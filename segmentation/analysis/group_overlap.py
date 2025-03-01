"""Script to compute the mIoU between group activations."""
import json
import os
from collections import defaultdict
from typing import Dict, List

import argh
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    PASCAL_CATEGORIES,
    PASCAL_ID_MAPPING,
    ADE20k_ID_2_LABEL,
)
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import log


@torch.no_grad()
def group_overlap(
    img: Image.Image,
    gt_ann: torch.Tensor,
    ppnet: PPNetMultiScale,
    class_proto_val: Dict[int, Dict[str, List[float]]],
    tot_proto_val: Dict[str, List[float]],
):
    """Compute the group activations union and intersection for the groups assigned to the same class"""

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    conv_features = ppnet.conv_features(img_tensor)
    _, _, patch_height, patch_width = conv_features.shape

    _, activations = ppnet.forward_from_conv_features(conv_features, return_activations=True)
    group_activation_list = ppnet.compute_group(activations)

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    for class_id in np.unique(gt_ann):

        if class_id == 0:
            continue

        group_activation = group_activation_list[class_id - 1]
        group_activation = group_activation.view(patch_height, patch_width, -1).detach().cpu().numpy()

        for i in range(ppnet.num_groups):

            g_act = group_activation[:, :, i]
            upsampled_g_act = cv2.resize(
                g_act, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )
            quant = np.quantile(upsampled_g_act, 0.95)
            bin_upsampled_g_act = (upsampled_g_act > quant).astype(int)

            for j in range(i + 1, ppnet.num_groups):

                g_act_j = group_activation[:, :, j]
                upsampled_g_act_j = cv2.resize(
                    g_act_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
                )
                quant_j = np.quantile(upsampled_g_act_j, 0.95)
                bin_upsampled_g_act_j = (upsampled_g_act_j > quant_j).astype(int)
                pr = bin_upsampled_g_act == 1
                gt = bin_upsampled_g_act_j == 1

                inter = np.sum(pr & gt)
                union = np.sum((pr | gt))

                class_proto_val[class_id - 1]["intersection"].append(inter)
                class_proto_val[class_id - 1]["union"].append(union)

                tot_proto_val["intersection"].append(inter)
                tot_proto_val["union"].append(union)

    return class_proto_val, tot_proto_val


def run_group_overlap(model_name: str, training_phase: str, data_type: str):
    """Wrapper function to loop through a dataset to compute the overlap across groups"""

    np.random.seed(42)
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)
    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

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

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    val_dataset = PatchClassificationDataset(
        data_type=data_type,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_margin_size=0,
        window_size=(513, 513),
        only_19_from_cityscapes=(data_type == "cityscapes"),
        scales=(0.5, 1.5),
        split_key="val",
        is_eval=True,
        push_prototypes=True,
    )

    class_proto_val = {k: defaultdict(list) for k in range(ppnet.num_classes)}
    tot_proto_val = defaultdict(list)

    for sample_id in tqdm(range(len(val_dataset)), desc="Computing Group Overlap", total=len(val_dataset)):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        gt_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))
        if data_type != "ade":
            gt_ann = val_dataset.convert_targets(gt_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        class_proto_val, tot_proto_val = group_overlap(img, gt_ann, ppnet, class_proto_val, tot_proto_val)

    final_mIoU = {}
    for class_id in range(ppnet.num_classes):
        class_proto_val[class_id]["intersection"] = np.sum(class_proto_val[class_id]["intersection"])
        class_proto_val[class_id]["union"] = np.sum(class_proto_val[class_id]["union"])

        final_mIoU[cls2name[class_id]] = class_proto_val[class_id]["intersection"] / class_proto_val[class_id]["union"]

    tot_proto_val["intersection"] = np.sum(tot_proto_val["intersection"])
    tot_proto_val["union"] = np.sum(tot_proto_val["union"])
    final_mIoU["total"] = tot_proto_val["intersection"] / tot_proto_val["union"]

    with open(os.path.join(model_path, "group_mIoU.json"), "w") as f:
        json.dump(final_mIoU, f)


if __name__ == "__main__":
    argh.dispatch_command(run_group_overlap)
