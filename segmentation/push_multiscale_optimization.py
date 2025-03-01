"""Computationally optimized code for the pushing mechanism at multi-scale"""

import json
import os
import time
from typing import Callable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from helpers import find_continuous_high_activation_crop, makedir
from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    COCO_ID_2_LABEL,
    EM_ID_2_LABEL,
    PASCAL_CATEGORIES,
    PASCAL_ID_MAPPING,
    ADE20k_ID_2_LABEL,
)
from segmentation.data.dataset import PatchClassificationDataset, resize_label
from segmentation.model.model_multiscale import PPNetMultiScale

to_tensor = transforms.ToTensor()


@torch.no_grad()
def compute_distances(
    ppnet: PPNetMultiScale,
    dataset: PatchClassificationDataset,
    img: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    max_dist: float = 1e10,
    device: str = "cpu",
    void_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the minimum distances for all prototypes to pixels assigned to the same class.

    Args:
        ppnet (PPNetMultiScale): Input model to compute the distances.
        dataset (PatchClassificationDataset): Input dataset for target conversion.
        img (np.ndarray): Input image to compute the distances to the prototypes
        target (np.ndarray): Input target to filter out for the pixels.
        num_classes (int): Number of classes for one hot encoding.
        max_dist (float, optional): Maximum distance to extract minimum. Defaults to 1e10.
        device (str, optional): Device for distance inferences. Defaults to "cpu".
        void_class (Optional[int], optional): Class to ignore. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Minimum indices and values for all prototypes.
    """

    # Model inference
    ppnet.to(device)
    ppnet.eval()
    img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)
    _, distances = ppnet(img_tensor, return_activations=False)

    # Convert label
    if dataset.convert_targets is not None:
        target = dataset.convert_targets(target)
    target_tensor = resize_label(target, (distances.shape[3], distances.shape[2])).unsqueeze(0).to(device)

    # Compute one hot encoding
    class_identiy = ppnet.prototype_class_identity.T.clone().to(device)
    target_one_hot = F.one_hot(target_tensor, num_classes=num_classes if void_class is None else num_classes + 1).to(
        torch.float32
    )

    # Filter void class
    if void_class is not None:
        target_one_hot = torch.concat(
            [target_one_hot[:, :, :, :void_class], target_one_hot[:, :, :, void_class + 1 :]], dim=-1
        )

    # Compute minimum distances via "masking" pixels when not assigned to class
    target_one_hot = torch.matmul(target_one_hot, class_identiy)
    target_one_hot = max_dist * (1 - target_one_hot.permute(0, 3, 1, 2))

    masked_distances = distances + target_one_hot
    masked_distances = masked_distances.flatten(-2, -1)

    return masked_distances.min(dim=-1).indices, masked_distances.min(dim=-1).values


def min_across_dataset(
    dataset: PatchClassificationDataset,
    ppnet: PPNetMultiScale,
    num_classes: int,
    void_class: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute the minimum distances to all prototypes across all images.

    Args:
        dataset (PatchClassificationDataset): Input dataset to loop over all samples.
        ppnet (PPNetMultiScale): Input model to compute all distances.
        num_classes (int): Number of classes.
        void_class (Optional[int], optional): Class to ignore. Defaults to None.
        device (str, optional): Device for distance inference. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: Return the image ids with the pixel ids for the minimum for all prototypes across the dataset.
    """

    list_idx = []
    list_dist = []
    for _, img_id in tqdm(enumerate(dataset.img_ids), desc="Computing Minimums on all Train", total=len(dataset)):

        img_path = dataset.get_img_path(img_id)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # remove margins which were used for training
        margin_size = dataset.image_margin_size
        img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))

        gt_ann = np.load(os.path.join(dataset.annotations_dir, img_id + ".npy"))

        idx, dist = compute_distances(
            ppnet, dataset, img, gt_ann, num_classes=num_classes, void_class=void_class, device=device
        )
        list_dist.append(dist)
        list_idx.append(idx)

    tot_dist = torch.concat(list_dist, dim=0)

    return tot_dist.argmin(dim=0), list_idx


@torch.no_grad()
def global_min(
    proto_min_dist: torch.Tensor,
    list_min_patch: List[torch.Tensor],
    dataset: PatchClassificationDataset,
    ppnet: PPNetMultiScale,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Compute the closest patches encoded features for all prototypes.

    Args:
        proto_min_dist (torch.Tensor): Image Ids of the minimum distance for each prototype.
        list_min_patch (List[torch.Tensor]): List of ids for the minimum patch per image to each prototype.
        dataset (PatchClassificationDataset): Input dataset to get the image path.
        ppnet (PPNetMultiScale): Input model to compute the feature mapp.
        device (str, optional): Device for inference. Defaults to "cpu".

    Returns:
        List[torch.Tensor]: List of encoded features for each prototype closest prototypes.
    """

    n_prototypes_scale = ppnet.num_prototypes // ppnet.num_scales
    ppnet.to(device)
    global_min_fmap_patches = []

    for p in tqdm(range(ppnet.num_prototypes), desc="Computing Minimums on all Prototypes", total=ppnet.num_prototypes):

        current_scale = p // n_prototypes_scale

        img_id = dataset.img_ids[proto_min_dist[p]]
        img_path = dataset.get_img_path(img_id)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # remove margins which were used for training
        margin_size = dataset.image_margin_size
        img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))
        img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)

        conv_features = ppnet.conv_features(img_tensor)
        _, C, H, W = conv_features.shape
        conv_features = conv_features.view(ppnet.num_scales, int(C / ppnet.num_scales), H, W)

        flattened_index = list_min_patch[proto_min_dist[p]][:, p].item()
        i = flattened_index // conv_features.shape[-1]
        j = flattened_index % conv_features.shape[-1]

        global_min_fmap_patches.append(conv_features[current_scale, :, i : i + 1, j : j + 1].detach().cpu().numpy())

    return global_min_fmap_patches


def push_prototypes_multiscale(
    dataset: PatchClassificationDataset,
    prototype_network_parallel: PPNetMultiScale,
    prototype_layer_stride: int = 1,
    root_dir_for_saving_prototypes: Optional[os.PathLike] = None,
    epoch_number: Optional[int] = None,
    prototype_img_filename_prefix: Optional[str] = None,
    prototype_self_act_filename_prefix: Optional[str] = None,
    proto_bound_boxes_filename_prefix: Optional[str] = None,
    save_prototype_class_identity: bool = True,
    log: Callable = print,
    data_type: str = "cityscapes",
    prototype_activation_function_in_numpy: Optional[Callable] = None,
    device: str = "cpu",
):
    """Wrapper function to compute the prototypes.

    Args:
        dataset (PatchClassificationDataset): Dataset for the extraction of samples.
        prototype_network_parallel (PPNetMultiScale): Model with prototypes.
        prototype_layer_stride (int): DEPRECATED.
        root_dir_for_saving_prototypes (os.PathLike, optional): File path to save the prototypes. Defaults to None.
        epoch_number (int, optional): Numbe rof the epochs. Defaults to None.
        prototype_img_filename_prefix (str, optional): Prefix for saving the images. Defaults to None.
        prototype_self_act_filename_prefix (str, optional): Prefix for saving the activations. Defaults to None.
        proto_bound_boxes_filename_prefix (str, optional): Prefix for saving the bounding box. Defaults to None.
        save_prototype_class_identity (bool): Flag to specify the class of each prototype. Defaults to True.
        log (callable): Function for logging. Defaults to print.
        data_type (str): Data type were are computing the prototypes on. Defaults to cityscapes.
        prototype_activation_function_in_numpy (callable, optional): Function for the activations. Defaults to None.
        device (str, optional): Device for inference. Defaults to "cpu".
    """

    if data_type == "cityscapes" or data_type == "pascal":
        ID_MAPPING = PASCAL_ID_MAPPING if (data_type == "pascal") else CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = PASCAL_CATEGORIES if (data_type == "pascal") else CITYSCAPES_CATEGORIES

        cls2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        if data_type == "pascal":
            cls2name = {i: CATEGORIES[k + 1] for i, k in cls2name.items() if k < len(CATEGORIES) - 1}
        else:
            cls2name = {i: CATEGORIES[k] for i, k in cls2name.items()}
    elif data_type == "ade":
        cls2name = ADE20k_ID_2_LABEL
    elif data_type == "em":
        cls2name = EM_ID_2_LABEL
    elif data_type == "coco":
        cls2name = COCO_ID_2_LABEL
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if hasattr(prototype_network_parallel, "module"):
        prototype_network_parallel = prototype_network_parallel.module

    prototype_network_parallel.eval()
    log("\tpush")

    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.num_prototypes

    """
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    """
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    num_classes = prototype_network_parallel.num_classes

    # for model that ignores void class | TODO: Check if this is necessary
    if hasattr(prototype_network_parallel, "void_class") and not prototype_network_parallel.void_class:
        num_classes = num_classes + 1

    log("Computing minimum distances between prototypes and images...")
    proto_min_dist, tot_list_idx = min_across_dataset(
        dataset, prototype_network_parallel, num_classes, void_class=0, device=device
    )

    log("Computing global minimum for all prototypes...")
    global_min_fmap_patches = global_min(
        proto_min_dist, tot_list_idx, dataset, prototype_network_parallel, device=device
    )

    log(f"Storing prototypes...")
    update_prototypes_on_image(
        dataset,
        prototype_network_parallel,
        proto_min_dist,
        tot_list_idx,
        proto_rf_boxes,
        proto_bound_boxes,
        cls2name=cls2name,
        dir_for_saving_prototypes=proto_epoch_dir,
        prototype_img_filename_prefix=prototype_img_filename_prefix,
        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
        prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
        device=device,
    )

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(
            os.path.join(
                proto_epoch_dir, proto_bound_boxes_filename_prefix + "-receptive_field" + str(epoch_number) + ".npy"
            ),
            proto_rf_boxes,
        )
        np.save(
            os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + ".npy"),
            proto_bound_boxes,
        )

    log("\tExecuting push ...")
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    # de-duplicate prototypes
    _, unique_index = np.unique(prototype_update, axis=0, return_index=True)
    duplicate_idx = [i for i in range(prototype_network_parallel.num_prototypes) if i not in unique_index]

    log(f"Removing {len(duplicate_idx)} duplicate prototypes.")
    prototype_network_parallel.prune_prototypes(duplicate_idx)
    os.makedirs(root_dir_for_saving_prototypes, exist_ok=True)
    with open(os.path.join(root_dir_for_saving_prototypes, "unique_prototypes.json"), "w") as fp:
        json.dump([int(i) for i in sorted(unique_index)], fp)

    end = time.time()
    log("\tpush time: \t{0}".format(end - start))


@torch.no_grad()
def update_prototypes_on_image(
    dataset: PatchClassificationDataset,
    ppnet,
    proto_min_dist,
    list_min_patch,
    proto_rf_boxes,  # this will be updated
    proto_bound_boxes,  # this will be updated
    cls2name,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    prototype_activation_function_in_numpy=None,
    device="cpu",
):
    """Function re-used from push.py and the ProtoSeg repository."""

    prototype_shape = ppnet.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    num_scales = ppnet.num_scales
    n_prototypes_scale = n_prototypes // num_scales
    ppnet.to(device)

    for p in tqdm(range(ppnet.num_prototypes), desc="Storing all Prototypes", total=ppnet.num_prototypes):

        current_scale = p // n_prototypes_scale

        img_id = dataset.img_ids[proto_min_dist[p]]
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

        logits, distances = ppnet.forward_from_conv_features(conv_features)

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

        # get segmentation map
        logits = logits.permute(0, 3, 1, 2)
        logits_inter = torch.nn.functional.interpolate(
            logits, size=original_img_j.shape[:2], mode="bilinear", align_corners=False
        )
        logits_inter = logits_inter[0]
        pred = torch.argmax(logits_inter, dim=0).cpu().detach().numpy()

        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(ppnet.prototype_class_identity[p]).item()

        flattened_index = list_min_patch[proto_min_dist[p]][:, p].item()
        patch_i = flattened_index // conv_features.shape[-1]
        patch_j = flattened_index % conv_features.shape[-1]
        del conv_features

        # get the receptive field boundary of the image patch
        # that generates the representation
        # protoL_rf_info = ppnet.proto_layer_rf_info
        # rf_prototype_j = compute_rf_prototype((search_batch.shape[2], search_batch.shape[3]),
        # batch_argmin_proto_dist, protoL_rf_info)

        rf_start_h_index = int(patch_i * patch_height)
        rf_end_h_index = int(patch_i * patch_height + patch_height) + 1

        rf_start_w_index = int(patch_j * patch_width)
        rf_end_w_index = int(patch_j * patch_width + patch_width) + 1

        rf_prototype_j = [0, rf_start_h_index, rf_end_h_index, rf_start_w_index, rf_end_w_index]

        # crop out the receptive field
        rf_img_j = original_img_j[rf_prototype_j[1] : rf_prototype_j[2], rf_prototype_j[3] : rf_prototype_j[4], :]

        # save the prototype receptive field information
        proto_rf_boxes[p, 0] = rf_prototype_j[0] + proto_min_dist[p]
        proto_rf_boxes[p, 1] = rf_prototype_j[1]
        proto_rf_boxes[p, 2] = rf_prototype_j[2]
        proto_rf_boxes[p, 3] = rf_prototype_j[3]
        proto_rf_boxes[p, 4] = rf_prototype_j[4]
        if proto_rf_boxes.shape[1] == 6:
            proto_rf_boxes[p, 5] = target_class

        # find the highly activated region of the original image
        proto_dist_img_j = proto_dist_[:, :, p]
        if ppnet.prototype_activation_function == "log":
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
        elif ppnet.prototype_activation_function == "linear":
            proto_act_img_j = max_dist - proto_dist_img_j
        else:
            proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)

        upsampled_act_img_j = cv2.resize(
            proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
        )

        # high activation area = percentile 95 calculated for activation for all pixels
        threshold = np.percentile(upsampled_act_img_j, 95)

        # show activation map only on the ground truth class
        y_mask = img_y.cpu().detach().numpy() == (target_class + 1)
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

        # save the prototype boundary (rectangular boundary of highly activated region)
        proto_bound_boxes[p, 0] = proto_rf_boxes[p, 0]
        proto_bound_boxes[p, 1] = proto_bound_j[0]
        proto_bound_boxes[p, 2] = proto_bound_j[1]
        proto_bound_boxes[p, 3] = proto_bound_j[2]
        proto_bound_boxes[p, 4] = proto_bound_j[3]
        if proto_bound_boxes.shape[1] == 6 and img_y is not None:
            proto_bound_boxes[p, 5] = target_class

        """
        proto_rf_boxes and proto_bound_boxes column:
        0: image index in the entire dataset
        1: height start index
        2: height end index
        3: width start index
        4: width end index
        5: (optional) class identity
        """
        if dir_for_saving_prototypes is not None:
            cls_name = cls2name[target_class]
            dir_for_saving_prototypes_cls = os.path.join(dir_for_saving_prototypes, cls_name)
            os.makedirs(dir_for_saving_prototypes_cls, exist_ok=True)
            DPI = 100

            # save segmentation
            plt.figure(figsize=(img_width / DPI, img_height / DPI))
            plt.figure(figsize=(img_width / DPI, img_height / DPI))
            plt.imshow(original_img_j)

            plt.imshow(pred, alpha=0.7)
            plt.axis("off")
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(
                os.path.join(
                    dir_for_saving_prototypes_cls,
                    prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original_segmentation.png",
                )
            )
            plt.close()

            if prototype_self_act_filename_prefix is not None:
                # save the numpy array of the prototype self activation
                np.save(
                    os.path.join(
                        dir_for_saving_prototypes_cls,
                        prototype_self_act_filename_prefix + f"_scale_{current_scale}_{p}.npy",
                    ),
                    proto_act_img_j,
                )
            if prototype_img_filename_prefix is not None:
                # save the whole image containing the prototype as png

                # save the whole image containing the prototype as png
                plt.imsave(
                    os.path.join(
                        dir_for_saving_prototypes_cls,
                        prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original.png",
                    ),
                    original_img_j,
                    vmin=0.0,
                    vmax=1.0,
                )
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
                        prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original_with_box.png",
                    )
                )
                plt.close()

                # overlay (upsampled) self activation on original image and save the result
                rescaled_act_img_j_gt = upsampled_act_img_j_gt - np.amin(upsampled_act_img_j_gt)
                rescaled_act_img_j_gt = rescaled_act_img_j_gt / np.amax(rescaled_act_img_j_gt)

                heatmap_gt = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j_gt), cv2.COLORMAP_JET)
                heatmap_gt = np.float32(heatmap_gt) / 255
                heatmap_gt = heatmap_gt[..., ::-1]

                overlayed_original_img_j_gt = 0.5 * original_img_j + 0.3 * heatmap_gt
                plt.imsave(
                    os.path.join(
                        dir_for_saving_prototypes_cls,
                        prototype_img_filename_prefix
                        + f"_scale_{current_scale}_{p}-original_with_self_act_gt_only.png",
                    ),
                    overlayed_original_img_j_gt,
                    vmin=0.0,
                    vmax=1.0,
                )

                # overlay (upsampled) self activation on original image and save the result
                rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

                heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]

                overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                plt.imsave(
                    os.path.join(
                        dir_for_saving_prototypes_cls,
                        prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-original_with_self_act.png",
                    ),
                    overlayed_original_img_j,
                    vmin=0.0,
                    vmax=1.0,
                )

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
                        + f"_scale_{current_scale}_{p}-original_with_self_act_and_box.png",
                    )
                )
                plt.close()

                if img_y.ndim > 2:
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix + f"_scale_{current_scale}_{p}-receptive_field.png",
                        ),
                        rf_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    overlayed_rf_img_j = overlayed_original_img_j[
                        rf_prototype_j[1] : rf_prototype_j[2], rf_prototype_j[3] : rf_prototype_j[4]
                    ]
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes_cls,
                            prototype_img_filename_prefix
                            + f"_scale_{current_scale}_{p}-receptive_field_with_self_act.png",
                        ),
                        overlayed_rf_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )

                # save the prototype image (highly activated region of the whole image)
                plt.imsave(
                    os.path.join(
                        dir_for_saving_prototypes_cls, prototype_img_filename_prefix + f"_scale_{current_scale}_{p}.png"
                    ),
                    proto_img_j,
                    vmin=0.0,
                    vmax=1.0,
                )

                # save the prototype image (highly activated region of the whole image)
                plt.imsave(
                    os.path.join(
                        dir_for_saving_prototypes_cls,
                        prototype_img_filename_prefix + f"_scale_{current_scale}_{p}_gt.png",
                    ),
                    proto_img_j_gt,
                    vmin=0.0,
                    vmax=1.0,
                )
