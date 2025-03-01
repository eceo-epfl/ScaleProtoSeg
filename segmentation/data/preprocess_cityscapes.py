"""
Preprocesss cityscapes before training a segmentation model.
https://www.cityscapes-dataset.com/

how to run run:

python -m segmentation.data.preprocess_cityscapes {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_ID_2_LABEL
from segmentation.utils import add_margins_to_image

# SOURCE_DATA_PATH must point to the unpacked CityScapes "Fine" images and labels
# Download it from https://www.cityscapes-dataset.com/downloads/
# (gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip)

# Cityscapes images have dimensions 1024x2024 pixels
# We generate the biggest possible "windowed" margin of 512 for a square sliding window of size 1024x1024
load_dotenv()

MARGIN_SIZE = 0

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_CITY"]
TARGET_PATH = os.environ["DATA_PATH_CITY"]
LABELS_PATH = os.path.join(SOURCE_PATH, "gtFine/")
IMAGES_PATH = os.path.join(SOURCE_PATH, "leftImg8bit/")

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, f"img_with_margin_{MARGIN_SIZE}")

CATEGORY_2_TRAIN_ID = {cat: i for i, cat in enumerate(CITYSCAPES_CATEGORIES)}
ID_2_TRAIN_ID = {i: CATEGORY_2_TRAIN_ID[cat] for i, cat in CITYSCAPES_ID_2_LABEL.items()}
id_2_train_id_vec = np.vectorize(ID_2_TRAIN_ID.get)


def process_images_in_chunks(args):
    split_key, city_name, png_files = args
    chunk_img_ids = []

    split_dir = os.path.join(LABELS_PATH, split_key)
    city_dir = os.path.join(split_dir, city_name)

    for file in png_files:
        img_id = file.split("_gtFine_labelIds.png")[0]
        chunk_img_ids.append(img_id)

        # 1. Save labels
        with open(os.path.join(city_dir, file), "rb") as f:
            label_ids = np.array(Image.open(f).convert("RGB"))[:, :, 0]
        label_ids = id_2_train_id_vec(label_ids).astype(np.uint8)
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, f"{img_id}.npy"), label_ids)

        # 2. Save image (with mirrored margin)
        input_img_path = os.path.join(IMAGES_PATH, split_key, city_name, img_id + "_leftImg8bit.png")
        with open(input_img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img_with_margin = add_margins_to_image(img, MARGIN_SIZE)
        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + ".png")
        img_with_margin.save(output_img_path)

    return chunk_img_ids


def preprocess_cityscapes(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Using {len(CITYSCAPES_CATEGORIES)} object categories using {n_jobs} threads.")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {"train": [], "val": [], "test": []}

    for split_key in tqdm(["train", "val", "test"], desc="preprocessing images"):
        split_dir = os.path.join(LABELS_PATH, split_key)
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        for city_name in tqdm(os.listdir(split_dir), desc=split_key):
            print(f"Processing {split_key}/{city_name}...")
            city_dir = os.path.join(split_dir, city_name)
            city_files = np.asarray([file for file in os.listdir(city_dir) if file.endswith("labelIds.png")])
            n_chunks = int(np.ceil(len(city_files) / chunk_size))
            chunk_files = np.array_split(city_files, n_chunks)

            parallel_args = [(split_key, city_name, chunk) for chunk in chunk_files]

            pool = multiprocessing.Pool(n_jobs)
            prog_bar = tqdm(total=len(city_files), desc=f"{split_key}/{city_name}")

            for chunk_img_ids in pool.imap_unordered(process_images_in_chunks, parallel_args):
                img_ids[split_key] += chunk_img_ids
                prog_bar.update(len(chunk_img_ids))

            prog_bar.close()
            pool.close()

    with open(os.path.join(TARGET_PATH, "all_images.json"), "w") as fp:
        json.dump(img_ids, fp)


if __name__ == "__main__":
    argh.dispatch_command(preprocess_cityscapes)
