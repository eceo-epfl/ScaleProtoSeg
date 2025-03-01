"""Script to build folders represening the group composition for all model classes"""

import json
import os
import shutil

import argh
import torch

from segmentation.constants import (
    CITYSCAPES_19_EVAL_CATEGORIES,
    CITYSCAPES_CATEGORIES,
    PASCAL_CATEGORIES,
    PASCAL_ID_MAPPING,
    ADE20k_ID_2_LABEL,
)
from settings import log


def group_comp(model_name: str, proto_name: str, training_phase: str, data_type: str, threshold: float):
    """Callable to extract the group composition."""

    threshold = float(threshold)

    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)
    proto_path = os.path.join(os.environ["RESULTS_DIR"], proto_name, "prototypes")
    output_path = os.path.join(os.environ["RESULTS_DIR"], model_name, "group_composition")

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

    # Approximation for CITYSCAPES, ADE, PASCAL
    num_scales = ppnet.num_scales
    num_prototypes = ppnet.num_prototypes
    n_prototypes_scale = num_prototypes // num_scales

    for cls_i in range(len(ppnet.group_projection)):
        cls_name = cls2name[cls_i]
        list_proto_idx = torch.nonzero(ppnet.prototype_class_identity[:, cls_i]).flatten().tolist()

        for k in range(ppnet.num_groups):
            group_weight = ppnet.group_projection[cls_i].weight.data[k, :]
            non_zero_proto_id = torch.nonzero(group_weight).flatten().tolist()
            flag = "single-low" if torch.sum(group_weight > threshold) <= 1 else "mutliple"
            output_path_group = os.path.join(output_path, f"class_{cls2name[cls_i]}_group_{k}_{flag}")
            os.makedirs(output_path_group, exist_ok=True)

            proto_info = []

            for group_proto_id in non_zero_proto_id:
                real_proto_id = list_proto_idx[group_proto_id]
                proto_weight = group_weight[group_proto_id].item()
                current_scale = real_proto_id // n_prototypes_scale

                proto_info.append(
                    {"real_proto_id": real_proto_id, "proto_weight": proto_weight, "scale": current_scale}
                )

                org_dir_prototypes_cls = os.path.join(proto_path, cls_name)
                file_1 = os.path.join(
                    org_dir_prototypes_cls, f"prototype-img_scale_{current_scale}_{real_proto_id}-original_with_box.png"
                )
                file_2 = os.path.join(
                    org_dir_prototypes_cls,
                    f"prototype-img_scale_{current_scale}_{real_proto_id}-original_with_self_act_and_box.png",
                )
                file_3 = os.path.join(
                    org_dir_prototypes_cls, f"prototype-img_scale_{current_scale}_{real_proto_id}.png"
                )
                file_4 = os.path.join(
                    org_dir_prototypes_cls, f"prototype-img_scale_{current_scale}_{real_proto_id}_gt.png"
                )

                shutil.copy(file_1, output_path_group)
                shutil.copy(file_2, output_path_group)
                shutil.copy(file_3, output_path_group)
                shutil.copy(file_4, output_path_group)

            # Save proto_info as JSON
            with open(os.path.join(output_path_group, "proto_info.json"), "w") as json_file:
                json.dump(proto_info, json_file, indent=4)


if __name__ == "__main__":
    argh.dispatch_command(group_comp)
