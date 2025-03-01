"""Script to analyze the failure cases via group and prototype activation plots"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict

import argh
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    COCO_ID_2_LABEL,
    EM_ID_2_LABEL,
    PASCAL_CATEGORIES,
    PASCAL_ID_MAPPING,
    ADE20k_ID_2_LABEL,
)
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import data_path, log


@torch.no_grad()
def failure_cases(model_name: str, training_phase: str, data_type: str, threshold: float):
    """For low performing class over images this function computes prototype and group activations"""

    threshold = float(threshold)
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)
    dataset_path = data_path[data_type]
    output_path = os.path.join(model_path, "failures", training_phase)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])

    margin = 0
    img_dir = os.path.join(dataset_path, f"img_with_margin_{margin}/val")

    all_img_files = [p for p in os.listdir(img_dir) if p.endswith(".npy")]
    ann_dir = os.path.join(dataset_path, "annotations/val")

    if data_type == "cityscapes" or data_type == "pascal":
        ID_MAPPING = PASCAL_ID_MAPPING if (data_type == "pascal") else CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = PASCAL_CATEGORIES if (data_type == "pascal") else CITYSCAPES_CATEGORIES
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        if data_type == "pascal":
            pred2name = {i: CATEGORIES[k + 1] for i, k in pred2name.items() if k < len(CATEGORIES) - 1}
        else:
            pred2name = {i: CATEGORIES[k] for i, k in pred2name.items()}
        CLS_CONVERT = np.vectorize(ID_MAPPING.get)
    elif data_type == "ade":
        pred2name = ADE20k_ID_2_LABEL
    elif data_type == "em":
        pred2name = EM_ID_2_LABEL
    elif data_type == "coco":
        pred2name = COCO_ID_2_LABEL
    else:
        raise ValueError(f"Unknown data type {data_type}")

    RESULTS_DIR = os.path.join(model_path, f"evaluation/{training_phase}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    correct_pixels, total_pixels = 0, 0

    with torch.no_grad():
        for img_id, img_file in enumerate(tqdm(all_img_files, desc="evaluating")):

            CLS_I = Counter()
            CLS_U = Counter()

            img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)
            ann = np.load(os.path.join(ann_dir, img_file))

            if margin != 0:
                img = img[margin:-margin, margin:-margin]

            if data_type == "pascal":  # TODO: Weirdly high
                ann = CLS_CONVERT(ann)
                img_shape = (513, 513)
                img_tensor = transform(img)
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=img_shape, mode="bilinear", align_corners=False
                )[0]
            elif data_type == "cityscapes":
                ann = CLS_CONVERT(ann)
                img_shape = ann.shape
                img_tensor = transform(img)
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=img_shape, mode="bilinear", align_corners=False
                )[0]
            elif data_type == "ade":
                img_tensor = transform(img)
                img_tensor = transforms.Resize(size=512, interpolation=transforms.InterpolationMode.BILINEAR)(
                    img_tensor
                )
            elif data_type == "em":
                img_tensor = transform(img)
                assert img_tensor.shape == (3, 512, 512), "EM Data not at 512 size"
            elif data_type == "coco":
                img_shape = ann.shape
                img_tensor = transform(img)
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=img_shape, mode="bilinear", align_corners=False
                )[0]

            img_tensor = img_tensor.unsqueeze(0).cuda()
            raw_logits, raw_activations = ppnet.forward(img_tensor, return_activations=True)

            logits = raw_logits.permute(0, 3, 1, 2)
            logits = F.interpolate(logits, size=ann.shape, mode="bilinear", align_corners=False)[0]
            pred = torch.argmax(logits, dim=0).cpu().detach().numpy()

            correct_pixels += np.sum(((pred + 1) == ann) & (ann != 0))
            total_pixels += np.sum(ann != 0)

            for cls_i in np.unique(ann):
                cls_i -= 1  # To align with the rest of the code

                if cls_i < 0:
                    continue

                pr = pred == cls_i
                gt = ann == cls_i + 1

                CLS_I[cls_i] += np.sum(pr & gt)
                CLS_U[cls_i] += np.sum((pr | gt) & (ann != 0))  # ignore pixels where ground truth is void

                IoU = CLS_I[cls_i] / CLS_U[cls_i] if CLS_U[cls_i] > 0 else 0

                if IoU < threshold:
                    mask_class_id = ann == cls_i + 1
                    pred_class_id = pred[mask_class_id]
                    mispredictions = pred_class_id[(pred_class_id != cls_i)]  # Ignore Background Pascal
                    miss_class_id = Counter(mispredictions).most_common(1)[0][0]

                    if miss_class_id == 0 and data_type == "pascal":
                        continue

                    else:
                        log(f"Failure case: {img_file} - {pred2name[cls_i]} - IoU: {IoU:.4f}")
                        plot_activation_group(
                            img,
                            img_id,
                            ann,
                            raw_activations,
                            raw_logits,
                            ppnet,
                            pred2name,
                            cls_i + 1,
                            miss_class_id + 1,
                            output_path,
                        )
                        plot_activation_proto(
                            img,
                            img_id,
                            ann,
                            raw_activations,
                            raw_logits,
                            ppnet,
                            pred2name,
                            cls_i + 1,
                            miss_class_id + 1,
                            output_path,
                        )


# Code from sample_activations_group
@torch.no_grad()
def plot_activation_group(
    img: Image.Image,
    img_id: int,
    gt_ann: torch.Tensor,
    activations: torch.Tensor,
    logits: torch.Tensor,
    ppnet: PPNetMultiScale,
    cls2name: Dict[int, str],
    class_id: int,
    miss_class_id: int,
    output_path: os.PathLike,
):
    """Plot the group activations over an image for two specific classes"""

    os.makedirs(os.path.join(output_path, str(img_id), cls2name[class_id - 1]), exist_ok=True)
    output_path = os.path.join(output_path, str(img_id), cls2name[class_id - 1])

    group_activation_list = ppnet.compute_group(activations)

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    # get segmentation map
    logits = logits.permute(0, 3, 1, 2)
    _, _, patch_height, patch_width = logits.shape
    logits_inter = torch.nn.functional.interpolate(
        logits, size=original_img_j.shape[:2], mode="bilinear", align_corners=False
    )
    logits_inter = logits_inter[0]
    pred = torch.argmax(logits_inter, dim=0).cpu().detach().numpy()
    rgb_pred = plt.cm.viridis(pred / pred.max())[:, :, :3]
    mask = gt_ann != class_id
    rgb_pred[mask] = [0, 0, 0]

    pred_image = 0.3 * original_img_j + 0.7 * rgb_pred
    plt.imsave(os.path.join(output_path, f"plot_img_{img_id}_segmentation.png"), arr=pred_image, dpi=150)
    plt.close()

    plt.imsave(os.path.join(output_path, f"plot_img_{img_id}_original.png"), arr=original_img_j, dpi=150)
    plt.close()

    y_mask = gt_ann == class_id
    list_upsample_g_act = []
    group_weights = {}

    mis_mask = pred != miss_class_id - 1
    rgb_pred[mis_mask] = [0, 0, 0]

    pred_image = 0.3 * original_img_j + 0.7 * rgb_pred
    plt.imsave(
        os.path.join(output_path, f"plot_img_{img_id}_segmentation_misprediction_{cls2name[miss_class_id - 1]}.png"),
        arr=pred_image,
        dpi=150,
    )
    plt.close()

    group_max = defaultdict(list)
    group_min = defaultdict(list)

    for cls_i in [class_id, miss_class_id]:
        for i in range(ppnet.num_groups):
            group_activation = group_activation_list[cls_i - 1]
            group_activation = group_activation.view(patch_height, patch_width, -1).detach().cpu().numpy()
            group_weights["group_" + str(i)] = ppnet.last_layer_group.weight.data[
                cls_i - 1, (cls_i - 1) * ppnet.num_groups + i
            ].item()
            g_act = group_activation[:, :, i]
            upsampled_g_act = cv2.resize(
                g_act, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )

            list_upsample_g_act.append(
                upsampled_g_act
                * ppnet.last_layer_group.weight.data[cls_i - 1, (cls_i - 1) * ppnet.num_groups + i].item()
            )
            upsampled_g_act = upsampled_g_act * y_mask

            group_max[i].append(np.max(upsampled_g_act))
            group_min[i].append(np.min(upsampled_g_act))

    group_max = {k: np.max(v) for k, v in group_max.items()}
    group_min = {k: np.min(v) for k, v in group_min.items()}

    for cls_i in [class_id, miss_class_id]:
        for i in range(ppnet.num_groups):
            group_activation = group_activation_list[cls_i - 1]
            group_activation = group_activation.view(patch_height, patch_width, -1).detach().cpu().numpy()
            group_weights["group_" + str(i)] = ppnet.last_layer_group.weight.data[
                cls_i - 1, (cls_i - 1) * ppnet.num_groups + i
            ].item()
            g_act = group_activation[:, :, i]
            upsampled_g_act = cv2.resize(
                g_act, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )

            list_upsample_g_act.append(
                upsampled_g_act
                * ppnet.last_layer_group.weight.data[cls_i - 1, (cls_i - 1) * ppnet.num_groups + i].item()
            )
            upsampled_g_act = upsampled_g_act * y_mask

            rescaled_g_act = upsampled_g_act - group_min[i]
            rescaled_g_act = rescaled_g_act / group_max[i]

            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_g_act), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]

            overlayed_original_img_j = 0.5 * original_img_j + 0.5 * heatmap
            plt.imsave(
                os.path.join(output_path, f"plot_img_{img_id}_group_{i}_class_{cls2name[cls_i - 1]}.png"),
                arr=overlayed_original_img_j,
                dpi=150,
            )
            plt.close()

        json_file_path = os.path.join(output_path, f"{cls2name[cls_i - 1]}_group_weights.json")
        with open(json_file_path, "w") as json_file:
            json.dump(group_weights, json_file, indent=4)


# Code from sample_activations_prototype
@torch.no_grad()
def plot_activation_proto(
    img: Image.Image,
    img_id: int,
    gt_ann: torch.Tensor,
    activations: torch.Tensor,
    logits: torch.Tensor,
    ppnet: PPNetMultiScale,
    cls2name: Dict[int, str],
    class_id: int,
    miss_class_id: int,
    output_path: os.PathLike,
):
    """Plot the prototype activations over an image for two specific classes"""

    os.makedirs(os.path.join(output_path, str(img_id), cls2name[class_id - 1]), exist_ok=True)
    output_path = os.path.join(output_path, str(img_id), cls2name[class_id - 1])

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    logits = logits.permute(0, 3, 1, 2)
    _, _, patch_height, patch_width = logits.shape

    proto_max = defaultdict(list)
    proto_min = defaultdict(list)

    y_mask = gt_ann == class_id

    for cls_i in [class_id, miss_class_id]:

        proto_ids = ppnet.prototype_class_identity[:, cls_i - 1].flatten().nonzero()

        for p in proto_ids:

            proto_act_img_j = activations[:, p].view(patch_height, patch_width).detach().cpu().numpy()
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )

            upsampled_act_img_j = upsampled_act_img_j * y_mask

            proto_max[p.item()].append(np.max(upsampled_act_img_j))
            proto_min[p.item()].append(np.min(upsampled_act_img_j))

    proto_max = {k: np.max(v) for k, v in proto_max.items()}
    proto_min = {k: np.min(v) for k, v in proto_min.items()}

    for cls_i in [class_id, miss_class_id]:

        proto_ids = ppnet.prototype_class_identity[:, cls_i - 1].flatten().nonzero()

        for p in proto_ids:

            proto_act_img_j = activations[:, p].view(patch_height, patch_width).detach().cpu().numpy()
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )

            upsampled_act_img_j = upsampled_act_img_j * y_mask

            rescaled_act_img_j = upsampled_act_img_j - proto_min[p.item()]
            rescaled_act_img_j = rescaled_act_img_j / proto_max[p.item()]

            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]

            overlayed_original_img_j = 0.5 * original_img_j + 0.5 * heatmap

            plt.imsave(
                os.path.join(output_path, f"plot_img_{img_id}_proto_{p.item()}_class_{cls2name[cls_i - 1]}.png"),
                arr=overlayed_original_img_j,
                dpi=150,
            )
            plt.close()


if __name__ == "__main__":
    argh.dispatch_command(failure_cases)
