"""
Script for validation across all datasets: cityscapes, pascal, ade, em, coco.
It is an extension of the code in eval_valid.py from https://github.com/gmum/proto-segmentation.
It works for both ProtoSeg and ScaleProtoSeg.
"""

import json
import os
from collections import Counter

import argh
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from segmentation.data.dataset import resize_label
from settings import data_path, log


@torch.no_grad()
def run_evaluation(
    model_name: str, training_phase: str, batch_size: int = 2, data_type: str = "cityscapes", margin: int = 0
):
    """Function to run the ealuation on the validation sets.

    Args:
        model_name (str): Model name to load for evalustion
        training_phase (str): Phase of the model to load for evaluation.
        batch_size (int, optional): Batch size used during evaluation, certain datasets have not consistent H, W. Defaults to 2.
        data_type (str, optional): Data to evaluate on. Defaults to "cityscapes".
        margin (int, optional): NOT USED. Defaults to 0.

    Raises:
        ValueError: Specify the right data type.
    """

    # Loading the model
    model_path = os.path.join(os.environ["RESULTS_DIR"], model_name)
    dataset_path = data_path[data_type]

    if isinstance(batch_size, str):
        batch_size = int(batch_size)

    if training_phase == "pruned":
        checkpoint_path = os.path.join(model_path, "pruned/checkpoints/push_last.pth")
    elif training_phase == "final-group" or training_phase == "push_final":
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}.pth")
    else:
        checkpoint_path = os.path.join(model_path, f"checkpoints/{training_phase}_last.pth")

    log(f"Loading model from {checkpoint_path}")
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    # Loading the data and class mapping.
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])

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

    cls_prototype_counts = [Counter() for _ in range(len(pred2name))]
    proto_ident = ppnet.prototype_class_identity.cpu().detach().numpy()
    mean_top_k = np.zeros(proto_ident.shape[0], dtype=float)

    RESULTS_DIR = os.path.join(model_path, f"evaluation/{training_phase}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Compute histogram of distances between same class prototypes
    proto2cls = {}
    cls2protos = {c: [] for c in range(ppnet.num_classes)}

    for proto_num in range(proto_ident.shape[0]):
        cls = np.argmax(proto_ident[proto_num])
        proto2cls[proto_num] = cls
        cls2protos[cls].append(proto_num)

    PROTO2CLS = np.vectorize(proto2cls.get)

    protos = ppnet.prototype_vectors.squeeze()

    all_cls_distances = []

    for cls_i in range(ppnet.num_classes):
        cls_proto_ind = (proto_ident[:, cls_i] == 1).nonzero()[0]
        if len(cls_proto_ind) < 2:
            all_cls_distances.append(None)
            continue
        cls_protos = protos[cls_proto_ind]

        distances = torch.cdist(cls_protos, cls_protos)
        distances = distances + 10e6 * torch.triu(torch.ones_like(distances, device=cls_protos.device))
        distances = distances.flatten()
        distances = distances[distances < 10e6]

        distances = distances.cpu().detach().numpy()
        all_cls_distances.append(distances)

    if len(pred2name) <= 20:
        n_rows = 4
    elif len(pred2name) <= 25:
        n_rows = 5
    else:
        n_rows = 40  # Potentially raise this value for COCO and ADE deu to large number of prototypes.
    _, axes = plt.subplots(n_rows, 5, figsize=(15, 12))

    plt.suptitle(f"{model_name} ({training_phase})\nHistogram of distances between same-class prototypes")
    axes = axes.flatten()
    class_i = 0

    for class_i, class_name in pred2name.items():
        if all_cls_distances[class_i] is None:
            continue
        axes[class_i].hist(all_cls_distances[class_i], bins=10)
        d_min, d_avg, d_max = (
            np.min(all_cls_distances[class_i]),
            np.mean(all_cls_distances[class_i]),
            np.max(all_cls_distances[class_i]),
        )
        axes[class_i].set_title(f"{class_name}\nmin: {d_min:.2f} avg: {d_avg:.2f} max: {d_max:.2f}")

    for i in range(class_i + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "histogram_dist_same_class_prototypes.png"))

    # Compute the intersection and union over the whole validation set via inference.
    CLS_I = Counter()
    CLS_U = Counter()

    np.random.shuffle(all_img_files)

    n_batches = int(np.ceil(len(all_img_files) / batch_size))
    batched_img_files = np.array_split(all_img_files, n_batches)

    correct_pixels, total_pixels = 0, 0

    with torch.no_grad():
        for batch_img_files in tqdm(batched_img_files, desc="evaluating"):
            img_tensors = []
            anns = []

            for img_file in batch_img_files:
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
                elif data_type == "coco":  # Based on deeplabv2 code for coco | TODO: test 513*513
                    img_shape = ann.shape
                    img_tensor = transform(img)
                    img_tensor = torch.nn.functional.interpolate(
                        img_tensor.unsqueeze(0), size=img_shape, mode="bilinear", align_corners=False
                    )[0]

                anns.append(ann)
                img_tensors.append(img_tensor)

            img_tensors = torch.stack(img_tensors, dim=0).cuda()
            batch_logits, batch_distances = ppnet.forward(img_tensors)

            batch_logits = batch_logits.permute(0, 3, 1, 2)

            for sample_i in range(len(batch_img_files)):
                ann = anns[sample_i]
                logits = torch.unsqueeze(batch_logits[sample_i], 0)
                distances = torch.unsqueeze(batch_distances[sample_i], 0)

                logits = F.interpolate(logits, size=ann.shape, mode="bilinear", align_corners=False)[0]
                distances = F.interpolate(distances, size=ann.shape, mode="bilinear", align_corners=False)[0]

                nearest_proto = torch.argmin(distances, dim=0).cpu().detach().numpy()
                distances = distances.cpu().detach().numpy()
                pred = torch.argmax(logits, dim=0).cpu().detach().numpy()

                correct_pixels += np.sum(((pred + 1) == ann) & (ann != 0))
                total_pixels += np.sum(ann != 0)

                for cls_i in range(ppnet.num_classes):
                    pr = pred == cls_i
                    gt = ann == cls_i + 1

                    CLS_I[cls_i] += np.sum(pr & gt)
                    CLS_U[cls_i] += np.sum((pr | gt) & (ann != 0))  # ignore pixels where ground truth is void

                # calculate statistics of prototypes occurrences as nearest
                nearest_proto_cls = PROTO2CLS(nearest_proto)

                for class_i, class_name in pred2name.items():
                    is_class_proto = (pred == class_i) & (nearest_proto_cls == class_i)
                    for proto_i, proto_num in enumerate(cls2protos[class_i]):
                        cls_prototype_counts[class_i][proto_i] += np.sum(is_class_proto & (nearest_proto == proto_num))
                del is_class_proto

                # calculate top K nearest prototypes for random sample of pixels for speed
                n_random_pixels = 100

                rows = np.random.randint(distances.shape[1], size=n_random_pixels)
                cols = np.random.randint(distances.shape[2], size=n_random_pixels)

                sample_distances = distances[:, rows, cols]
                sample_preds = pred[rows, cols]

                nearest_pixel_protos = np.argsort(sample_distances, axis=0)
                is_class_proto = PROTO2CLS(nearest_pixel_protos) == sample_preds

                for k in range(nearest_pixel_protos.shape[0]):
                    nearest_k = np.sum(is_class_proto[: k + 1]) / (k + 1)
                    mean_top_k[k] += nearest_k * 100 / n_random_pixels

    # Compute the metrics.
    pixel_accuracy = correct_pixels / total_pixels * 100

    CLS_IOU = {cls_i: (CLS_I[cls_i] * 100) / u for cls_i, u in CLS_U.items() if u > 0}
    mean_iou = np.mean(list(CLS_IOU.values()))
    log(f"{model_name} {training_phase} mIOU: {mean_iou}")

    keys = list(sorted(CLS_IOU.keys()))

    vals = [CLS_IOU[k] for k in keys]
    keys = [pred2name[cls_i] for cls_i in keys]

    plt.figure(figsize=(15, 5))
    xticks = np.arange(len(keys))
    plt.bar(xticks, vals)
    plt.xticks(xticks, keys, rotation=45)
    plt.title(
        f"{model_name} ({training_phase})\nIOU scores over all {len(CLS_IOU)} classes (mean IOU: {mean_iou:.4f}, pixel accuracy: {pixel_accuracy:.4f})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "iou_scores.png"))

    with open(os.path.join(RESULTS_DIR, "iou_scores.json"), "w") as fp:
        json.dump(CLS_IOU, fp)

    with open(os.path.join(RESULTS_DIR, "mean_iou.txt"), "w") as fp:
        fp.write(str(mean_iou))

    # Analysis of the nearest prototypes per pixel
    plt.figure(figsize=(10, 5))
    plt.title(
        f"{model_name} ({training_phase})\nHow many of the nearest K prototypes to a random pixel are from its predicted class?"
    )
    plt.xlabel("Nearest K prototypes to a pixel")
    plt.ylabel("% of K prototypes from pixel class")
    plt.ylim([0, 100])
    xticks = [i for i in range(proto_ident.shape[0]) if i % 10 == 0]
    plt.xticks(xticks, xticks)
    plt.plot(mean_top_k / (len(batched_img_files) * batch_size))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_prototypes_in_nearest_k.png"))

    if len(pred2name) <= 20:
        n_rows = 4
    elif len(pred2name) <= 25:
        n_rows = 5
    else:
        n_rows = 30  # Potentially raise this value for COCO and ADE deu to large number of prototypes.
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 12))
    plt.suptitle(
        f"{model_name} ({training_phase})\nOccurences (%) of 10 prototypes of each class in its top nearest class for each pixel"
    )
    axes = axes.flatten()
    for class_i, class_name in pred2name.items():
        n, c = zip(*cls_prototype_counts[class_i].most_common())
        if sum(cls_prototype_counts[class_i].values()) > 0:
            c = c / sum(cls_prototype_counts[class_i].values()) * 100
        axes[class_i].bar(np.arange(len(c)), c)
        axes[class_i].set_xticks(np.arange(len(c)), n)
        axes[class_i].set_title(class_name)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "occurences_of_each_class_prototypes_in_nearest_pixel.png"))

    # run the following code to visualize on some samples
    N_SAMPLES = 5
    DPI = 100

    for example_i, img_file in tqdm(
        enumerate(np.random.choice(all_img_files, size=N_SAMPLES, replace=False)),
        total=N_SAMPLES,
        desc="nearest prototype visualization",
    ):
        img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)

        ann = np.load(os.path.join(ann_dir, img_file))

        if data_type == "cityscapes" or data_type == "pascal":
            ann = CLS_CONVERT(ann)
        if data_type == "pascal" or data_type == "coco":  # Visualize 513x513
            ann = resize_label(ann, size=(513, 513)).cpu().detach().numpy()
            img_shape = (513, 513)
        elif data_type == "ade" or data_type == "em":
            ann = resize_label(ann, size=(512, 512)).cpu().detach().numpy()
            img_shape = (512, 512)
        else:
            if margin != 0:
                img = img[margin:-margin, margin:-margin]
            img_shape = (img.shape[0], img.shape[1])

        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).cuda()
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=img_shape, mode="bilinear", align_corners=False
            )
            logits, distances = ppnet.forward(img_tensor)

            img = torch.tensor(img).cuda().permute(2, 0, 1).unsqueeze(0).float()
            img = torch.nn.functional.interpolate(img, size=img_shape, mode="bilinear", align_corners=False)
            img = img.cpu().detach().numpy()[0].astype(int)
            img = img.transpose(1, 2, 0)

            logits = logits.permute(0, 3, 1, 2)

            logits = F.interpolate(logits, size=img_shape, mode="bilinear", align_corners=False)[0]
            distances = F.interpolate(distances, size=img_shape, mode="bilinear", align_corners=False)[0]

            # (H, W, C)
            distances = distances.cpu().detach().numpy()
            logits = logits.cpu().detach().numpy()

        nearest_proto = np.argmin(distances, axis=0)
        pred = np.argmax(logits, axis=0)

        # save some RAM
        del distances, logits, img_tensor

        void_mask = (ann == 0).astype(float)

        plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI))
        plt.title(f"{model_name} ({training_phase})\nExample {example_i}. Prediction (from interpolated logits)")
        plt.imshow(img)
        plt.imshow(pred, alpha=0.5)
        plt.imshow(np.zeros_like(pred), alpha=void_mask, vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"example_{example_i}_prediction.png"))
        plt.close()

        plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI))
        plt.title(
            f"{model_name} ({training_phase})\nExample {example_i}. Nearest prototypes (from interpolated distances)"
        )
        plt.imshow(img)
        plt.imshow(nearest_proto, alpha=0.5, vmin=0, vmax=9)
        plt.imshow(np.zeros_like(pred), alpha=void_mask, vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(RESULTS_DIR, f"example_{example_i}_prototypes.png"))
        plt.close()


if __name__ == "__main__":
    argh.dispatch_command(run_evaluation)
