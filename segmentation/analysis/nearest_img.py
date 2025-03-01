"""Inspired by the pushing code for multiscale compute the nearest images to each prototype"""

import os
import shutil

import argh
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from helpers import find_continuous_high_activation_crop
from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    PASCAL_CATEGORIES,
    PASCAL_ID_MAPPING,
    ADE20k_ID_2_LABEL,
)
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model_multiscale import PPNetMultiScale
from segmentation.utils import non_zero_proto
from settings import log

to_tensor = transforms.ToTensor()


@torch.no_grad()
def min_across_dataset(dataset: PatchClassificationDataset, ppnet: PPNetMultiScale, device: str = "cpu", n: int = 3):
    """Compute top n closest patches to each prototype."""

    list_idx = []
    list_dist = []
    for _, img_id in tqdm(enumerate(dataset.img_ids), desc="Computing Minimums", total=len(dataset)):

        img_path = dataset.get_img_path(img_id)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # remove margins which were used for training
        margin_size = dataset.image_margin_size
        img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))

        # Computing distances
        ppnet.to(device)
        ppnet.eval()
        img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)
        _, distances = ppnet(img_tensor, return_activations=False)

        distances = distances.flatten(-2, -1)
        idx, dist = distances.min(dim=-1).indices, distances.min(dim=-1).values
        list_dist.append(dist)
        list_idx.append(idx)

    tot_dist = torch.concat(list_dist, dim=0)
    _, top_n_idx = tot_dist.topk(n, dim=0, largest=False)

    return top_n_idx, list_idx


def nearest_img(model_name: str, training_phase: str, data_type: str, split_key: str, group_name: str):
    """Wrapper function to loop through a dataset and plot the closest images to a prototype."""

    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)
    model_group_path = os.path.join(os.environ["RESULTS_DIR"], group_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
        # checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    group_checkpoint_path = os.path.join(model_group_path, f"checkpoints/th-0.05-nopush-group_last.pth")

    output_path = os.path.join(model_path, f"nearest_img/{training_phase}")
    os.makedirs(output_path, exist_ok=True)

    if data_type == "cityscapes" or data_type == "pascal":

        ID_MAPPING = PASCAL_ID_MAPPING if (data_type == "pascal") else CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = PASCAL_CATEGORIES if (data_type == "pascal") else CITYSCAPES_CATEGORIES

        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        if data_type == "pascal":
            cls2name = {i: CATEGORIES[k + 1] for i, k in pred2name.items() if k < len(CATEGORIES) - 1}
        else:
            cls2name = {i: CATEGORIES[k] for i, k in pred2name.items()}

    else:
        cls2name = ADE20k_ID_2_LABEL

    cls2name[-1] = "void"

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    log(f"Loading group model from {group_checkpoint_path}")
    group_ppnet = torch.load(group_checkpoint_path)
    group_ppnet = group_ppnet.cuda()
    group_ppnet.eval()
    non_zero_proto_list = non_zero_proto(group_ppnet)
    print(non_zero_proto_list)

    dataset = PatchClassificationDataset(
        data_type=data_type,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_margin_size=0,
        window_size=(513, 513),
        only_19_from_cityscapes=(data_type == "cityscapes"),
        scales=(0.5, 1.5),
        split_key=split_key,
        is_eval=True,
        push_prototypes=True,
    )

    proto_min_dist, tot_list_idx = min_across_dataset(dataset, ppnet, device=device)

    log(f"Storing prototypes...")
    plot_prototypes_on_image(
        dataset,
        ppnet,
        proto_min_dist,
        tot_list_idx,
        cls2name=cls2name,
        org_dir_prototypes=os.path.join(model_path, "prototypes"),
        dir_for_saving_prototypes=output_path,
        prototype_img_filename_prefix="prototype-img",
        non_zero_proto_list=non_zero_proto_list,
        device=device,
    )


@torch.no_grad()
def plot_prototypes_on_image(
    dataset: PatchClassificationDataset,
    ppnet,
    proto_min_dist,
    list_min_patch,
    cls2name,
    org_dir_prototypes=None,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    non_zero_proto_list=None,
    device="cpu",
):
    """Function re-used with some modifications from push.py and the ProtoSeg repository."""

    prototype_shape = ppnet.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    # This is an approximation valid for ADE, CITY, PASCAL
    num_scales = ppnet.num_scales
    n_prototypes_scale = n_prototypes // num_scales
    ppnet.to(device)

    if non_zero_proto_list is None:
        non_zero_proto_list = list(range(ppnet.num_prototypes))

    for p in tqdm(range(ppnet.num_prototypes), desc="Storing all Prototypes", total=ppnet.num_prototypes):

        flag_zero = "used" if p in non_zero_proto_list else "unused"

        current_scale = p // n_prototypes_scale  # This is an approximation valid for ADE, CITY, PASCAL
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(ppnet.prototype_class_identity[p]).item()
        cls_name = cls2name[target_class]

        if org_dir_prototypes is not None:
            org_dir_prototypes_cls = os.path.join(org_dir_prototypes, cls_name)
            dir_for_saving_prototypes_cls = os.path.join(
                dir_for_saving_prototypes, cls_name, f"{str(p)}_{flag_zero}", "org"
            )
            os.makedirs(dir_for_saving_prototypes_cls, exist_ok=True)

            file_1 = os.path.join(
                org_dir_prototypes_cls,
                prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original_with_box.png",
            )
            file_2 = os.path.join(
                org_dir_prototypes_cls,
                prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original_with_self_act_and_box.png",
            )
            file_3 = os.path.join(
                org_dir_prototypes_cls, prototype_img_filename_prefix + f"_scale_{current_scale}_{p}.png"
            )
            file_4 = os.path.join(
                org_dir_prototypes_cls, prototype_img_filename_prefix + f"_scale_{current_scale}_{p}_gt.png"
            )

            shutil.copy(file_1, dir_for_saving_prototypes_cls)
            shutil.copy(file_2, dir_for_saving_prototypes_cls)
            shutil.copy(file_3, dir_for_saving_prototypes_cls)
            shutil.copy(file_4, dir_for_saving_prototypes_cls)

        for top_k, min_id in enumerate(proto_min_dist[:, p].tolist()):

            img_id = dataset.img_ids[min_id]
            img_path = dataset.get_img_path(img_id)

            with open(img_path, "rb") as f:
                img = Image.open(f).convert("RGB")

            # remove margins which were used for training
            margin_size = dataset.image_margin_size
            img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))
            img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)

            conv_features = ppnet.conv_features(img_tensor)
            del img_tensor

            img_y = np.load(os.path.join(dataset.annotations_dir, img_id + ".npy"))
            if dataset.convert_targets is not None:  # TBD
                img_y = dataset.convert_targets(img_y)
            img_y = torch.LongTensor(img_y)

            _, distances = ppnet.forward_from_conv_features(conv_features)

            model_output_height = conv_features.shape[2]
            model_output_width = conv_features.shape[3]

            img_height = img_y.shape[0]
            img_width = img_y.shape[1]

            patch_height = img_height / model_output_height
            patch_width = img_width / model_output_width

            proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()
            del distances

            # get the whole image
            original_img_j = to_tensor(img).detach().cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_height = original_img_j.shape[0]
            original_img_width = original_img_j.shape[1]

            flattened_index = list_min_patch[min_id][:, p].item()
            patch_i = flattened_index // conv_features.shape[-1]
            patch_j = flattened_index % conv_features.shape[-1]
            del conv_features

            rf_start_h_index = int(patch_i * patch_height)
            rf_end_h_index = int(patch_i * patch_height + patch_height) + 1

            rf_start_w_index = int(patch_j * patch_width)
            rf_end_w_index = int(patch_j * patch_width + patch_width) + 1

            rf_prototype_j = [0, rf_start_h_index, rf_end_h_index, rf_start_w_index, rf_end_w_index]

            patch_class = (
                img_y[rf_start_h_index:rf_end_h_index, rf_start_w_index:rf_end_w_index].flatten().cpu().detach().numpy()
            )
            patch_class = np.bincount(patch_class).argmax()
            patch_class -= 1  # Align to target class name

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[:, :, p]
            if ppnet.prototype_activation_function == "log":
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
            elif ppnet.prototype_activation_function == "linear":
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                raise ValueError("Unknown activation function")

            upsampled_act_img_j = cv2.resize(
                proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )

            # high activation area = percentile 95 calculated for activation for all pixels
            threshold = np.percentile(upsampled_act_img_j, 95)

            # show activation map only on the ground truth class
            y_mask = img_y.cpu().detach().numpy() == (patch_class + 1)
            upsampled_act_img_j_gt = upsampled_act_img_j * y_mask

            proto_bound_j = find_continuous_high_activation_crop(
                upsampled_act_img_j_gt, rf_prototype_j[1:], threshold=threshold
            )
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0] : proto_bound_j[1], proto_bound_j[2] : proto_bound_j[3], :]

            # find high activation *only* on ground truth
            if np.sum(y_mask) == 0:
                threshold_gt = np.inf
            else:
                threshold_gt = np.percentile(upsampled_act_img_j[y_mask], 95)

            proto_bound_j_gt = find_continuous_high_activation_crop(
                upsampled_act_img_j_gt, rf_prototype_j[1:], threshold=threshold_gt
            )
            # crop out the image patch with high activation as prototype image
            proto_img_j_gt = original_img_j[
                proto_bound_j_gt[0] : proto_bound_j_gt[1], proto_bound_j_gt[2] : proto_bound_j_gt[3], :
            ]

            if dir_for_saving_prototypes is not None:

                dir_for_saving_prototypes_cls = os.path.join(
                    dir_for_saving_prototypes, cls_name, f"{str(p)}_{flag_zero}", "closest"
                )
                os.makedirs(dir_for_saving_prototypes_cls, exist_ok=True)
                DPI = 100

                if prototype_img_filename_prefix is not None:
                    plt.figure(figsize=(img_width / DPI, img_height / DPI))
                    plt.imshow(original_img_j)
                    plt.plot(
                        [rf_start_w_index, rf_start_w_index],
                        [rf_start_h_index, rf_end_h_index],
                        [rf_end_w_index, rf_end_w_index],
                        [rf_start_h_index, rf_end_h_index],
                        [rf_start_w_index, rf_end_w_index],
                        [rf_start_h_index, rf_start_h_index],
                        [rf_start_w_index, rf_end_w_index],
                        [rf_end_h_index, rf_end_h_index],
                        linewidth=2,
                        color="red",
                    )
                    plt.axis("off")
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix
                            + f"_class_{cls2name[patch_class]}_scale_{current_scale}_{p}_closest_{top_k}-original_with_box.png",
                        )
                    )
                    plt.close()

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j_gt = upsampled_act_img_j_gt - np.amin(upsampled_act_img_j_gt)
                    rescaled_act_img_j_gt = rescaled_act_img_j_gt / np.amax(rescaled_act_img_j_gt)

                    heatmap_gt = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j_gt), cv2.COLORMAP_JET)
                    heatmap_gt = np.float32(heatmap_gt) / 255
                    heatmap_gt = heatmap_gt[..., ::-1]

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]

                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.figure(figsize=(img_width / DPI, img_height / DPI))
                    plt.imshow(overlayed_original_img_j)
                    plt.plot(
                        [rf_start_w_index, rf_start_w_index],
                        [rf_start_h_index, rf_end_h_index],
                        [rf_end_w_index, rf_end_w_index],
                        [rf_start_h_index, rf_end_h_index],
                        [rf_start_w_index, rf_end_w_index],
                        [rf_start_h_index, rf_start_h_index],
                        [rf_start_w_index, rf_end_w_index],
                        [rf_end_h_index, rf_end_h_index],
                        linewidth=2,
                        color="red",
                    )
                    plt.axis("off")
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix
                            + f"_class_{cls2name[patch_class]}_scale_{current_scale}_{p}_closest_{top_k}-original_with_self_act_and_box.png",
                        )
                    )
                    plt.close()

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix
                            + f"_class_{cls2name[patch_class]}_scale_{current_scale}_{p}_closest_{top_k}.png",
                        ),
                        proto_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix
                            + f"_class_{cls2name[patch_class]}_scale_{current_scale}_{p}_closest_{top_k}_gt.png",
                        ),
                        proto_img_j_gt,
                        vmin=0.0,
                        vmax=1.0,
                    )


if __name__ == "__main__":
    argh.dispatch_command(nearest_img)
