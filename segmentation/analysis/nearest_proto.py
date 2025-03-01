"""Inspired by the pushing code for multiscale compute the nearest prototypes to each image"""
import os
import argh
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from helpers import find_continuous_high_activation_crop
from settings import log
from find_nearest import to_normalized_tensor
from torchvision import transforms
import shutil
from typing import List, Optional

from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model_multiscale import PPNetMultiScale
from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, CITYSCAPES_CATEGORIES,\
    PASCAL_CATEGORIES, PASCAL_ID_MAPPING, ADE20k_ID_2_LABEL
from segmentation.utils import non_zero_proto


to_tensor = transforms.ToTensor()


@torch.no_grad()
def min_on_sample(dataset: PatchClassificationDataset, ppnet: PPNetMultiScale, sample_id: int, device: str = 'cpu', n: int = 3, include_list: Optional[List[int]]=None):
    """For a given sample select n random pixels and extract the closest prototypes to each pixel."""

    img_id = dataset.img_ids[sample_id]
    img_path = dataset.get_img_path(img_id)

    if include_list is None:
        include_list = list(range(ppnet.num_prototypes))

    gt_ann = np.load(os.path.join(dataset.annotations_dir, img_id + '.npy'))
    if dataset.convert_targets is not None:
        gt_ann = dataset.convert_targets(gt_ann)

    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    # remove margins which were used for training
    margin_size = dataset.image_margin_size
    img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))

    # Computing distances
    ppnet.to(device)
    ppnet.eval()
    img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)
    _, distances = ppnet(img_tensor, return_activations=False)

    _, _, H, W = distances.shape
    indices_H = torch.randint(H, (n,))
    indices_W = torch.randint(W, (n,))

    sample_idx = []

    for h, w in zip(indices_H, indices_W):
        _, min_idx = distances[:, :, h, w].topk(n + 10, largest=False)
        min_idx = min_idx.flatten().tolist()
        min_idx = [idx for idx in min_idx if idx in include_list][:n]
        sample_idx.append(min_idx)

    return sample_idx, (indices_H, indices_W)


def nearest_proto(model_name: str, group_name: str, training_phase: str, data_type: str,
                  split_key: str, n_samples: int = None):
    """Wrapper to compute the nearest prototypes to random pixels in a set of samples."""

    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    model_group_path = os.path.join(os.environ['RESULTS_DIR'], group_name)
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(42)

    n_samples = int(n_samples) if n_samples is not None else None

    if training_phase == 'pruned':
        checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
        #checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
        checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    group_checkpoint_path = os.path.join(model_group_path, f'checkpoints/th-0.05-nopush-group_last.pth')

    output_path = os.path.join(model_path, f'nearest_proto/{training_phase}')
    os.makedirs(output_path, exist_ok=True)

    if data_type == "cityscapes" or data_type == "pascal":

        ID_MAPPING = PASCAL_ID_MAPPING if (data_type == "pascal") else CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = PASCAL_CATEGORIES if (data_type == "pascal") else CITYSCAPES_CATEGORIES

        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        if (data_type == "pascal"):
            cls2name = {i: CATEGORIES[k+1] for i, k in pred2name.items() if k < len(CATEGORIES)-1}
        else:
            cls2name = {i: CATEGORIES[k] for i, k in pred2name.items()}

    else:
        cls2name = ADE20k_ID_2_LABEL


    log(f'Loading model from {checkpoint_path}')
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    log(f'Loading group model from {group_checkpoint_path}')
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
            only_19_from_cityscapes = (data_type == "cityscapes"),
            scales = (0.5, 1.5),
            split_key=split_key,
            is_eval=True,
            push_prototypes=True
        )

    if n_samples is not None:
        sample_ids = np.random.choice(len(dataset), n_samples, replace=False)
    else:
        sample_ids = np.arange(len(dataset))


    for sample_id in tqdm(sample_ids, desc='Computing Minimum & Plotting', total=len(sample_ids)):

        tot_list_idx, indices = min_on_sample(dataset, ppnet, sample_id, device=device, include_list=non_zero_proto_list)

        log(f'Storing prototypes ... for sample {sample_id}')
        update_prototypes_on_image(dataset, ppnet, sample_id, indices, tot_list_idx,
                                   cls2name=cls2name, org_dir_prototypes=os.path.join(model_path, 'prototypes'),
                                   dir_for_saving_prototypes=output_path, prototype_img_filename_prefix='prototype-img',
                                   device=device)


@torch.no_grad()
def update_prototypes_on_image(dataset: PatchClassificationDataset,
                               ppnet,
                               sample_id,
                               indices,
                               list_min_patch,
                               cls2name,
                               org_dir_prototypes=None,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               device='cpu'):
    """Function re-used with some modifications from push.py and the ProtoSeg repository."""

    prototype_shape = ppnet.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    num_scales = ppnet.num_scales
    n_prototypes_scale = n_prototypes // num_scales
    ppnet.to(device)

    img_id = dataset.img_ids[sample_id]
    img_path = dataset.get_img_path(img_id)

    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    # remove margins which were used for training
    margin_size = dataset.image_margin_size
    img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))
    img_tensor = to_normalized_tensor(img).unsqueeze(0).to(device)

    conv_features = ppnet.conv_features(img_tensor)
    del img_tensor

    img_y = np.load(os.path.join(dataset.annotations_dir, img_id + '.npy'))
    if dataset.convert_targets is not None: # TBD
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

    output_bound = {}
    output_color = ["red", "green", "blue"]

    for k, (h, w) in enumerate(zip(*indices)):

        prototypes = list_min_patch[k]
        print(prototypes)
        for clo_idx, p in enumerate(tqdm(prototypes, desc='Storing all Prototypes', total=len(prototypes))):

            current_scale = p // n_prototypes_scale
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(ppnet.prototype_class_identity[p]).item()
            cls_name = cls2name[target_class]

            if org_dir_prototypes is not None:
                org_dir_prototypes_cls = os.path.join(org_dir_prototypes, cls_name)
                dir_for_saving_prototypes_cls = os.path.join(dir_for_saving_prototypes, str(sample_id), f"{cls_name}_loc_{k}")
                os.makedirs(dir_for_saving_prototypes_cls, exist_ok=True)

                file_2 = os.path.join(org_dir_prototypes_cls,
                                        prototype_img_filename_prefix + f'_scale_{current_scale}_{p}-original_with_self_act_and_box.png')
                shutil.copy(file_2, dir_for_saving_prototypes_cls)


            patch_i = h
            patch_j = w

            rf_start_h_index = int(patch_i * patch_height)
            rf_end_h_index = int(patch_i * patch_height + patch_height) + 1

            rf_start_w_index = int(patch_j * patch_width)
            rf_end_w_index = int(patch_j * patch_width + patch_width) + 1

            rf_prototype_j = [0, rf_start_h_index, rf_end_h_index, rf_start_w_index, rf_end_w_index]

            patch_class = img_y[rf_start_h_index:rf_end_h_index, rf_start_w_index:rf_end_w_index].flatten().cpu().detach().numpy()
            patch_class = np.bincount(patch_class).argmax()
            patch_class -= 1 # Align to target class name

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[:, :, p]
            if ppnet.prototype_activation_function == 'log':
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
            elif ppnet.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                raise ValueError('Unknown activation function')

            threshold = proto_act_img_j[patch_i, patch_j].item() * 0.99
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_width, original_img_height),
                                                interpolation=cv2.INTER_CUBIC)

            # show activation map only on the ground truth class
            y_mask = img_y.cpu().detach().numpy() == (patch_class + 1)
            upsampled_act_img_j_gt = upsampled_act_img_j * y_mask

            proto_bound_j = find_continuous_high_activation_crop(upsampled_act_img_j_gt, rf_prototype_j[1:],
                                                                 threshold=threshold)

            output_bound[k] = rf_prototype_j[1:]
                                                                 
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                            proto_bound_j[2]:proto_bound_j[3], :]

            if dir_for_saving_prototypes is not None:

                DPI = 100

                if prototype_img_filename_prefix is not None:

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

                    plt.plot([proto_bound_j[2], proto_bound_j[2]], [proto_bound_j[0], proto_bound_j[1]],
                                [proto_bound_j[3], proto_bound_j[3]], [proto_bound_j[0], proto_bound_j[1]],
                                [proto_bound_j[2], proto_bound_j[3]], [proto_bound_j[0], proto_bound_j[0]],
                                [proto_bound_j[2], proto_bound_j[3]], [proto_bound_j[1], proto_bound_j[1]],
                                linewidth=2, color='red')

                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                                f'sample_{sample_id}_loc_{k}_closest_{clo_idx}_scale_{current_scale}_{p}-original_with_self_act_and_box.png'))
                    plt.close()

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes_cls,
                                            prototype_img_filename_prefix + f'_{sample_id}_loc_{k}_closest_{clo_idx}_scale_{current_scale}_{p}.png'),
                                proto_img_j,
                                vmin=0.0,
                                vmax=1.0)

    plt.figure(figsize=(img_width / DPI, img_height / DPI))
    plt.imshow(original_img_j)

    for k, proto_bound_j in output_bound.items():
        plt.plot([proto_bound_j[2], proto_bound_j[2]], [proto_bound_j[0], proto_bound_j[1]],
                [proto_bound_j[3], proto_bound_j[3]], [proto_bound_j[0], proto_bound_j[1]],
                [proto_bound_j[2], proto_bound_j[3]], [proto_bound_j[0], proto_bound_j[0]],
                [proto_bound_j[2], proto_bound_j[3]], [proto_bound_j[1], proto_bound_j[1]],
                linewidth=2, color=output_color[k])

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    sample_path = os.path.join(dir_for_saving_prototypes, str(sample_id))
    plt.savefig(os.path.join(sample_path,
                              f'sample_{sample_id}_scale_{current_scale}_{p}-original_with_box.png'))
    plt.close()

if __name__ == '__main__':
    argh.dispatch_command(nearest_proto)