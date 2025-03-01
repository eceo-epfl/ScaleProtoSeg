"""Script to plot the prototype activations over random samples."""

import os
from typing import Dict, Union

import argh
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
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
from segmentation.model.model import PPNet
from segmentation.model.model_multiscale import PPNetMultiScale
from settings import log


@torch.no_grad()
def plot_activation_proto(
    img: Image.Image,
    img_id: int,
    gt_ann: torch.Tensor,
    ppnet: Union[PPNetMultiScale, PPNet],
    cls2name: Dict[int, str],
    output_path: os.PathLike,
):
    """Function to plot the prototype activations."""

    cmap = ListedColormap(["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "gray", "cyan"])

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    conv_features = ppnet.conv_features(img_tensor)
    logits, distances = ppnet.forward_from_conv_features(conv_features)
    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()

    original_img_j = transforms.ToTensor()(img).detach().cpu().numpy()
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
    rgb_pred = plt.cm.viridis(pred / pred.max())[:, :, :3]
    mask = gt_ann == 0
    rgb_pred[mask] = [0, 0, 0]

    pred_image = 0.3 * original_img_j + 0.7 * rgb_pred
    plt.imsave(os.path.join(output_path, f"plot_img_{img_id}_segmentation.png"), arr=pred_image, dpi=150)
    plt.close()

    plt.imsave(os.path.join(output_path, f"plot_img_{img_id}_original.png"), arr=original_img_j, dpi=150)
    plt.close()

    for class_id in np.unique(gt_ann):

        if class_id == 0:
            continue

        os.makedirs(os.path.join(output_path, cls2name[class_id - 1]), exist_ok=True)

        proto_ids = ppnet.prototype_class_identity[:, class_id - 1].flatten().nonzero()
        list_upsample_g_act = []
        y_mask = gt_ann == class_id

        for p in proto_ids:

            proto_dist_img_j = proto_dist_[:, :, p]
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))

            upsampled_act_img_j = cv2.resize(
                proto_act_img_j, dsize=(original_img_width, original_img_height), interpolation=cv2.INTER_CUBIC
            )
            list_upsample_g_act.append(upsampled_act_img_j)

            upsampled_act_img_j = upsampled_act_img_j * y_mask

            rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
            rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]

            overlayed_original_img_j = 0.5 * original_img_j + 0.5 * heatmap

            plt.imsave(
                os.path.join(
                    output_path,
                    cls2name[class_id - 1],
                    f"plot_img_{img_id}_proto_{p.item()}_class_{cls2name[class_id - 1]}.png",
                ),
                arr=overlayed_original_img_j,
                dpi=150,
            )
            plt.close()

        if len(list_upsample_g_act) == 0:
            print(f"No prototypes for class {cls2name[class_id - 1]}")
            print(f"Proto ids: {proto_ids}")

            continue
        tot_g_act = np.stack(list_upsample_g_act, axis=-1)
        arg_tot_g_act = np.argmax(tot_g_act, axis=-1)

        rgb_g_act = cmap(arg_tot_g_act)[:, :, :3]
        rgb_g_act = rgb_g_act * y_mask[:, :, np.newaxis]

        overlayed_original_img_j = 0.5 * original_img_j + 0.5 * rgb_g_act
        plt.imsave(
            os.path.join(
                output_path, cls2name[class_id - 1], f"plot_img_{img_id}_max_class_{cls2name[class_id - 1]}.png"
            ),
            arr=overlayed_original_img_j,
            dpi=150,
        )
        plt.close()


def run_sample_activation(model_name: str, training_phase: str, data_type: str):
    """Wrapper function to sample n images for plotting the prototype activations."""

    np.random.seed(42)

    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)

    if training_phase == "pruned":
        # checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
        checkpoint_path = os.path.join(model_path, "pruned/pruned.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    output_path = os.path.join(model_path, f"prototype_activations/{training_phase}")
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

    sample_ids = np.random.choice(len(val_dataset), 200, replace=False)

    for sample_id in tqdm(sample_ids, desc="Plotting Activations", total=len(sample_ids)):

        img_id = val_dataset.img_ids[sample_id]
        img_path = val_dataset.get_img_path(img_id)

        gt_ann = np.load(os.path.join(val_dataset.annotations_dir, img_id + ".npy"))

        if data_type != "ade":
            gt_ann = val_dataset.convert_targets(gt_ann)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        plot_activation_proto(img, img_id, gt_ann, ppnet, cls2name, output_path)


if __name__ == "__main__":
    argh.dispatch_command(run_sample_activation)
