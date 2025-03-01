"""Script to compute the mIoU between prototype activations"""

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
)
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model_multiscale import PPNetMultiScale
from settings import log


@torch.no_grad()
def prototype_overlap(
    img: Image.Image,
    gt_ann: torch.Tensor,
    ppnet: PPNetMultiScale,
    class_proto_val: Dict[int, Dict[str, List[float]]],
    tot_proto_val: Dict[str, List[float]],
):
    """Compute the interesection and union of prototype activations for prototypes assigned to the same class"""

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    conv_features = ppnet.conv_features(img_tensor)
    logits, distances = ppnet.forward_from_conv_features(conv_features)
    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    for class_id in np.unique(gt_ann):

        if class_id == 0:
            continue

        proto_ids = ppnet.prototype_class_identity[:, class_id - 1].flatten().nonzero()

        for j in range(len(proto_ids)):

            p = proto_ids[j]

            proto_dist_img_j = proto_dist_[:, :, p]
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))

            upsampled_act_img_j = cv2.resize(
                proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )
            quant = np.quantile(upsampled_act_img_j, 0.95)
            bin_upsampled_g_act = (upsampled_act_img_j > quant).astype(int)

            for k in range(j + 1, len(proto_ids)):

                g = proto_ids[k]

                proto_dist_img_k = proto_dist_[:, :, g]
                proto_act_img_k = np.log((proto_dist_img_k + 1) / (proto_dist_img_k + ppnet.epsilon))

                upsampled_act_img_k = cv2.resize(
                    proto_act_img_k, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
                )
                quant_k = np.quantile(upsampled_act_img_k, 0.95)
                bin_upsampled_g_act_k = (upsampled_act_img_k > quant_k).astype(int)
                pr = bin_upsampled_g_act == 1
                gt = bin_upsampled_g_act_k == 1

                inter = np.sum(pr & gt)
                union = np.sum((pr | gt))

                class_proto_val[class_id - 1]["intersection"].append(inter)
                class_proto_val[class_id - 1]["union"].append(union)

                tot_proto_val["intersection"].append(inter)
                tot_proto_val["union"].append(union)

    return class_proto_val, tot_proto_val


def run_proto_activation(model_name: str, training_phase: str, pascal: bool = False):
    """Wrapper function to loop over validation dataset to compute mIoU between prototypes"""

    np.random.seed(42)

    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    CATEGORIES = PASCAL_CATEGORIES if pascal else CITYSCAPES_CATEGORIES

    cls2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
    if pascal:
        cls2name = {i: CATEGORIES[k + 1] for i, k in cls2name.items() if k < len(CATEGORIES) - 1}
    else:
        cls2name = {i: CATEGORIES[k] for i, k in cls2name.items()}

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    val_dataset = PatchClassificationDataset(
        data_type="cityscapes" if not pascal else "pascal",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_margin_size=0,
        window_size=(513, 513),
        only_19_from_cityscapes=not pascal,
        scales=(0.5, 1.5),
        split_key="val",
        is_eval=True,
        push_prototypes=True,
    )

    class_proto_val = {k: defaultdict(list) for k in range(ppnet.num_classes)}
    tot_proto_val = defaultdict(list)
    sample_ids = np.random.choice(len(val_dataset), 500, replace=False)

    for sample_id in tqdm(sample_ids, desc="Plotting Activations", total=len(sample_ids)):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        gt_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))
        gt_ann = val_dataset.convert_targets(gt_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        class_proto_val, tot_proto_val = prototype_overlap(img, gt_ann, ppnet, class_proto_val, tot_proto_val)

    final_mIoU = {}
    for class_id in range(ppnet.num_classes):
        class_proto_val[class_id]["intersection"] = np.sum(class_proto_val[class_id]["intersection"])
        class_proto_val[class_id]["union"] = np.sum(class_proto_val[class_id]["union"])

        final_mIoU[cls2name[class_id]] = class_proto_val[class_id]["intersection"] / class_proto_val[class_id]["union"]

    tot_proto_val["intersection"] = np.sum(tot_proto_val["intersection"])
    tot_proto_val["union"] = np.sum(tot_proto_val["union"])
    final_mIoU["total"] = tot_proto_val["intersection"] / tot_proto_val["union"]

    with open(os.path.join(model_path, "prototype_mIoU.json"), "w") as f:
        json.dump(final_mIoU, f)


if __name__ == "__main__":
    argh.dispatch_command(run_proto_activation)
