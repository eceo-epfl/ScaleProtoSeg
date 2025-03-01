"""
Dataset for training prototype patch classification model on Cityscapes, PASCAL, ADE20K, COCO-Stuff, and EM datasets
"""

import json
import os
import random
from typing import Any, List, Optional, Tuple

import cv2
import gin
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, PASCAL_ID_MAPPING
from settings import data_path, log


def resize_label(label, size):
    """
    Downsample labels by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    label = Image.fromarray(label.astype(float)).resize(size, resample=Image.NEAREST)
    return torch.LongTensor(np.asarray(label))


@gin.configurable(allowlist=["data_type", "mean", "std", "image_margin_size", "window_size", "scales", "jitter", "only_19_from_cityscapes"])
class PatchClassificationDataset(VisionDataset):
    def __init__(
        self,
        split_key: str,
        is_eval: bool,
        push_prototypes: bool = False,
        data_type: str = gin.REQUIRED,
        mean: List[float] = gin.REQUIRED,
        std: List[float] = gin.REQUIRED,
        image_margin_size: int = gin.REQUIRED,
        window_size: Optional[Tuple[int, int]] = None,
        only_19_from_cityscapes: bool = False,
        scales: Tuple[float] = (1.0,),
        jitter: bool = False,
    ):
        """Initialize the PatchClassificationDataset

        Args:
            split_key (str): Split to load (train, val, test)
            is_eval (bool): Whether to load the dataset for evaluation -> no shuffling in DataLoader
            push_prototypes (bool, optional): _description_. Defaults to False.
            data_type (str, optional): String representing the data used. Defaults to gin.REQUIRED.
            mean (List[float], optional): Mean values for the normalization of RGB inputs. Defaults to gin.REQUIRED.
            std (List[float], optional): Std values for the normalization of RGB inputs. Defaults to gin.REQUIRED.
            image_margin_size (int, optional): Integer representing the margin used in preprocessing. Defaults to gin.REQUIRED.
            window_size (Optional[Tuple[int, int]], optional): Window size used for training. Defaults to None.
            only_19_from_cityscapes (bool): NOT USED. Defaults to False.
            scales (Tuple[float], optional): Scales factor for augmentation. Defaults to (1.0,).
            jitter (bool, optional): Flag for input image jittering. Defaults to False.
        """
        self.mean = mean
        self.std = std
        self.is_eval = is_eval
        self.split_key = split_key
        self.annotations_dir = os.path.join(data_path[data_type], "annotations", split_key)
        self.push_prototypes = push_prototypes
        self.image_margin_size = image_margin_size
        self.window_size = window_size
        self.scales = scales

        # Define the mapping for the target labels if necessary
        if data_type == "cityscapes":
            self.convert_targets = np.vectorize(CITYSCAPES_19_EVAL_CATEGORIES.get)
        elif data_type == "pascal":
            self.convert_targets = np.vectorize(PASCAL_ID_MAPPING.get)
        else:  # ADE, EM, COCO already has proper IDs (COCO because of preprocessing)
            self.convert_targets = None

        # Load Image paths
        self.img_dir = os.path.join(data_path[data_type], f"img_with_margin_{self.image_margin_size}/{split_key}")

        # Define the transform
        if push_prototypes:
            transform = None

        elif jitter:
            transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.Normalize(mean, std),
                ]
            )

        else:
            transform = transforms.Compose([transforms.Normalize(mean, std)])

        # Initialize the vision dataset
        super(PatchClassificationDataset, self).__init__(root=self.img_dir, transform=transform)

        # Load the image ids
        with open(os.path.join(data_path[data_type], "all_images.json"), "r") as fp:
            self.img_ids = json.load(fp)[split_key]
        self.img_id2idx = {img_id: i for i, img_id in enumerate(self.img_ids)}

        log(f"Loaded {len(self.img_ids)} samples from {split_key} set")

    def __len__(self) -> int:
        return len(self.img_ids)

    def get_img_path(self, img_id: str) -> str:
        return os.path.join(self.img_dir, img_id + ".png")

    def __getitem__(self, index: int) -> Any:
        """Get the image and label for the given index"""

        # Load the image and label
        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, img_id + ".npy")
        label_path = os.path.join(self.annotations_dir, img_id + ".npy")
        image = np.load(img_path).astype(np.uint8)
        label = np.load(label_path)

        if self.window_size is None:
            window_size = label.shape
        else:
            window_size = self.window_size

        if label.ndim == 3:
            label = label[:, :, 0]

        if self.convert_targets is not None:
            label = self.convert_targets(label)
        label = label.astype(np.int32)

        if self.image_margin_size != 0:
            image = image[
                self.image_margin_size : -self.image_margin_size, self.image_margin_size : -self.image_margin_size
            ]

        h, w = label.shape

        if len(self.scales) < 2:
            scale_factor = 1.0
        else:
            scale_factor = random.uniform(self.scales[0], self.scales[1])
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # [0-255] to [0-1]
        image = image / 255.0

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(window_size[0] - h, 0)
        pad_w = max(window_size[1] - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

        # Cropping
        h, w = label.shape

        start_h = random.randint(0, h - window_size[0])
        start_w = random.randint(0, w - window_size[1])
        end_h = start_h + window_size[0]
        end_w = start_w + window_size[1]
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

        # HWC -> CHW
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        label = torch.tensor(label)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
