"""
Preprocesss PASCAL VOC 2012 dataset before training a segmentation model.
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

how to run run:

python -m segmentation.data.preprocess_pascal {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_PASCAL"]
TARGET_PATH = os.environ["DATA_PATH_PASCAL"]

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, "img_with_margin_0")


def process_images_in_chunks(args):
    split_key, img_ids = args
    chunk_img_ids = []

    unique_classes = set()

    for img_id in img_ids:
        chunk_img_ids.append(img_id)

        # 1. Save labels
        if split_key != "test":
            with open(os.path.join(SOURCE_PATH, f"SegmentationClassAug/{img_id}.png"), "rb") as f:
                img = Image.open(f).convert("RGB")

            pix = np.array(img).astype(np.uint8)
            pix = pix[:, :, 0]
            unique_classes.update(set(np.unique(pix)))
            np.save(os.path.join(ANNOTATIONS_DIR, split_key, img_id), pix)

        # 2. Save image
        input_img_path = os.path.join(SOURCE_PATH, f"JPEGImages/{img_id}.jpg")

        with open(input_img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + ".png")
        img.save(output_img_path)

        # Save image as .npy for fast loading
        pix = np.array(img).astype(np.uint8)
        np.save(os.path.join(MARGIN_IMG_DIR, split_key, img_id), pix)

    return chunk_img_ids, unique_classes


def preprocess_pascal(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Preprocessing PASCAL VOC 2012")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {"train_aug": [], "train": [], "val": [], "test": []}

    split_info_dir = os.path.join(SOURCE_PATH, "ImageSets/SegmentationAug")

    for split_key in tqdm(["train_aug", "train", "val"], desc="preprocessing images"):  # TEST
        split_img_ids = [
            img_id.strip().split("/")[-1].split(".")[0]
            for img_id in open(os.path.join(split_info_dir, f"{split_key}.txt"), "r")
        ]

        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        n_chunks = int(np.ceil(len(split_img_ids) / chunk_size))
        chunk_files = np.array_split(split_img_ids, n_chunks)

        parallel_args = [(split_key, chunk) for chunk in chunk_files]

        pool = multiprocessing.Pool(n_jobs)
        prog_bar = tqdm(total=len(split_img_ids), desc=f"{split_key}")

        unique_classes = set()

        for chunk_img_ids, chunk_classes in pool.imap_unordered(process_images_in_chunks, parallel_args):
            img_ids[split_key] += chunk_img_ids
            unique_classes.update(set(chunk_classes))
            prog_bar.update(len(chunk_img_ids))

        prog_bar.close()
        pool.close()

        print(f"{split_key} unique classes:", unique_classes)

    with open(os.path.join(TARGET_PATH, "all_images.json"), "w") as fp:
        json.dump(img_ids, fp)


if __name__ == "__main__":
    argh.dispatch_command(preprocess_pascal)
