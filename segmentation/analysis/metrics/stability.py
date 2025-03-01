"""Script to evaluate models for stability"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import argh
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from find_nearest import to_normalized_tensor
from segmentation.analysis.equivariance import quantile_map
from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    MAX_PARTS_CITY,
    MAX_PARTS_PASCAL,
    PASCAL_CATEGORIES,
    PASCAL_FILTER_CLASS,
    PASCAL_ID_MAPPING,
)
from segmentation.data.dataset import PatchClassificationDataset
from segmentation.model.model import PPNet
from segmentation.model.model_multiscale_group import PPNetMultiScale
from settings import log


def proto_filter(ppnet: PPNetMultiScale) -> List[int]:
    """Callable to filter prototypes when not used in grouping mechanism."""
    tot_list_proto_ids = []
    for class_id in range(len(ppnet.group_projection)):
        proto_ids = torch.nonzero(ppnet.group_projection[class_id].weight.data.sum(dim=0))
        proto_ids = proto_ids.flatten().tolist()
        class_proto_ids = ppnet.prototype_class_identity[:, class_id].flatten().nonzero()
        class_proto_ids = class_proto_ids.flatten().tolist()
        filter_class_proto_ids = [class_proto_ids[proto_id] for proto_id in proto_ids]
        tot_list_proto_ids.extend(filter_class_proto_ids)
    return tot_list_proto_ids


def run_stability(
    model_name: str,
    training_phase: str,
    data_type: str,
    quantile: float = 0.8,
    std_dev: float = 0.2,
    group_name: str = None,
):
    """Wrapper to compute the stability across validation sets

    Args:
        model_name (str): Name of the model for testing.
        training_phase (str):  Training phase for loading the model.
        data_type (str): Data type for testing: Cityscapes & Pascal.
        quantile (float, optional): Quantile to threshold the prototype activations. Defaults to 0.8.
        std_dev (float, optional): Standard deviation of the noise injected. Defaults to 0.2.
        group_name (str, optional): Name of the model when grouping is used. Defaults to None.

    Raises:
        NotImplementedError: Data type not supported.
    """

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
        # checkpoint_path = os.path.join(model_path, 'pruned/pruned.pth')
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    output_path = os.path.join(model_path, f"metrics/{training_phase}")
    os.makedirs(output_path, exist_ok=True)

    global MAX_PARTS

    if data_type == "cityscapes":  # TODO: Potentially adapt to new mapping

        ID_MAPPING = CITYSCAPES_19_EVAL_CATEGORIES
        CATEGORIES = CITYSCAPES_CATEGORIES
        MAX_PARTS = MAX_PARTS_CITY
        FILTER_CLASS = []
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k] for i, k in pred2name.items()}

    elif data_type == "pascal":
        ID_MAPPING = PASCAL_ID_MAPPING
        CATEGORIES = PASCAL_CATEGORIES
        MAX_PARTS = MAX_PARTS_PASCAL
        FILTER_CLASS = PASCAL_FILTER_CLASS
        pred2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
        cls2name = {i: CATEGORIES[k + 1] for i, k in pred2name.items() if k < len(CATEGORIES) - 1}

    else:
        raise NotImplementedError(f"Data type {data_type} not supported")

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    if group_name is not None:
        group_path = os.path.join(os.environ["RESULTS_DIR"], group_name)
        checkpoint_path_group = os.path.join(group_path, f"checkpoints/th-0.05-nopush-group_last.pth")
        ppnet_group = torch.load(checkpoint_path_group)
        ppnet_group.eval()
        proto_ids = proto_filter(ppnet_group)
    else:
        proto_ids = None

    val_dataset = PatchClassificationDataset(  # TODO: Adapt to new dataset
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

    i = 0
    tot_presence = []
    tot_presence_noise = []

    for sample_id in tqdm(range(len(val_dataset)), desc="Computing Stability Score Validation", total=len(val_dataset)):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        path_cls = Path(val_dataset.annotations_dir)
        path_part = str(path_cls.parent) + f"_PIDS/{path_cls.name}"

        # Attention for PASCAL only overlap
        if not os.path.exists(os.path.join(path_part, img_id + ".npy")):
            continue

        part_ann = np.load(os.path.join(path_part, img_id + ".npy"))
        cls_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))
        cls_ann = val_dataset.convert_targets(cls_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        _, part_presence_list = part_intersect(img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids)
        _, noise_part_presence_list = part_intersect(
            img, cls_ann, part_ann, ppnet, cls2name, img_id, quantile, proto_ids, std_dev
        )

        tot_presence.extend(part_presence_list)
        tot_presence_noise.extend(noise_part_presence_list)
        i += 1

    df_presence = pd.DataFrame(
        tot_presence, columns=[f"part_{i}" for i in range(MAX_PARTS + 1)] + ["proto_id", "class", "img_id"]
    )
    df_presence_noise = pd.DataFrame(
        tot_presence_noise, columns=[f"part_stab_{i}" for i in range(MAX_PARTS + 1)] + ["proto_id", "class", "img_id"]
    )

    df_tot = df_presence.merge(df_presence_noise, on=["proto_id", "class", "img_id"])
    for i in range(MAX_PARTS + 1):
        df_tot[f"eq_part_{i}"] = (df_tot[f"part_{i}"] == df_tot[f"part_stab_{i}"]).astype(int)
    df_tot["sum_eq"] = df_tot[[f"eq_part_{i}" for i in range(MAX_PARTS + 1)]].sum(axis=1)
    df_tot["is_stable"] = (df_tot["sum_eq"] == MAX_PARTS + 1).astype(int)
    df_tot.to_csv(os.path.join(output_path, f"tot_presence_stab_qt_{quantile}.csv"), index=False)
    stab_score = df_tot.is_stable.mean()
    log(f"Stability score: {stab_score}")

    with open(os.path.join(output_path, f"stability_score_qt_{quantile}.txt"), "w") as f:
        f.write(f"{stab_score}")


@torch.no_grad()
def part_intersect(
    img: Image.Image,
    cls_ann: torch.Tensor,
    part_ann: torch.Tensor,
    ppnet: Union[PPNetMultiScale, PPNet],
    cls2name: Dict[int, str],
    img_id: int,
    quantile: float,
    noise: Optional[float] = None,
    filter_class_ids: List[int] = [],
) -> Tuple[Dict[int, List[int]], List[List[Any]]]:
    """Compute intersection between part labels and prototype activations."""

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()

    if noise is not None:
        noise = noise * torch.randn_like(img_tensor)
        img_tensor = img_tensor + noise
    conv_features = ppnet.conv_features(img_tensor)
    _, distances = ppnet.forward_from_conv_features(conv_features)
    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    part_presence = {}
    part_presence_list = []

    for class_id in np.unique(cls_ann):

        if class_id == 0:
            continue

        if class_id in filter_class_ids:
            continue

        proto_ids = ppnet.prototype_class_identity[:, class_id - 1].flatten().nonzero()
        if filter_proto_ids is not None:
            proto_ids = [p for p in proto_ids if p in filter_proto_ids]
        y_mask = cls_ann == class_id

        part_centroids = {}
        part_mask = part_ann * y_mask

        for part_id in np.unique(part_mask):
            if part_id <= 0:
                continue
            bin_part_mask = (part_mask == part_id).astype(np.uint8)
            _, _, _, centroids = cv2.connectedComponentsWithStats(bin_part_mask, 8, cv2.CV_32S)
            centroids = np.round(centroids).astype(int)
            part_centroids[part_id] = centroids

        if len(part_centroids) == 0:
            continue

        else:
            for p in proto_ids:
                presence_list = [0] * (MAX_PARTS + 1)  # Can keep 0 here instead of np.nan
                proto_dist_img_j = proto_dist_[:, :, p]
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))

                upsampled_act_img_j = cv2.resize(
                    proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_NEAREST
                )
                upsampled_act_img_j = upsampled_act_img_j * y_mask
                upsampled_act_img_j = upsampled_act_img_j[None, :, :]
                upsampled_act_img_j = quantile_map(torch.tensor(upsampled_act_img_j), quantile).numpy()

                for part_id, centroids in part_centroids.items():
                    for centroid in centroids:
                        x, y = centroid[0], centroid[1]
                        presence_list[part_id] += upsampled_act_img_j[y, x, 0]
                part_presence[p.item()] = [0 if val == 0 else 1 for val in presence_list]
                part_presence_list.append(part_presence[p.item()] + [p.item(), cls2name[class_id - 1], img_id])

    return part_presence, part_presence_list


if __name__ == "__main__":
    argh.dispatch_command(run_stability)
