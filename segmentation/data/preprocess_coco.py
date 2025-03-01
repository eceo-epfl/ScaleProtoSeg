"""
Preprocess COCO data.
https://github.com/nightrome/cocostuff?tab=readme-ov-file#downloads

how to run run:

python -m segmentation.data.preprocess_coco {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import COCO_ID_2_LABEL, COCO_ID_MAPPING
from segmentation.utils import add_margins_to_image

load_dotenv()

MARGIN_SIZE = 0

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_COCO"]
TARGET_PATH = os.environ["DATA_PATH_COCO"]
LABELS_PATH = os.path.join(SOURCE_PATH, "annotations_png/")
IMAGES_PATH = os.path.join(SOURCE_PATH, "images/")

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, f"img_with_margin_{MARGIN_SIZE}")
id_2_train_id_vec = np.vectorize(COCO_ID_MAPPING.get)


def process_images_in_chunks(args):
    split_key, png_files = args
    chunk_img_ids = []

    split_dir = os.path.join(LABELS_PATH, split_key)

    for file in png_files:
        img_id = file.split(".png")[0]
        chunk_img_ids.append(img_id)

        # 1. Save labels
        with open(os.path.join(split_dir, file), "rb") as f:
            label_ids = np.array(Image.open(f).convert("RGB"))[:, :, 0]
        label_ids = id_2_train_id_vec(label_ids).astype(np.uint8)
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, f"{img_id}.npy"), label_ids)

        # 2. Save image (with mirrored margin)
        input_img_path = os.path.join(IMAGES_PATH, split_key, img_id + ".jpg")
        with open(input_img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img_with_margin = add_margins_to_image(img, MARGIN_SIZE)
        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + ".png")  # TODO: Check if .jpeg is necessary
        img_with_margin.save(output_img_path)

    return chunk_img_ids


def preprocess_coco(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Using {len(COCO_ID_2_LABEL)} object categories using {n_jobs} threads.")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {"train": [], "val": []}

    for split_key in tqdm(["train", "val"], desc="preprocessing images"):
        split_dir = os.path.join(LABELS_PATH, split_key)
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        print(f"Processing {split_key}")
        img_files = np.asarray([file for file in os.listdir(split_dir) if file.endswith(".png")])
        n_chunks = int(np.ceil(len(img_files) / chunk_size))
        chunk_files = np.array_split(img_files, n_chunks)

        parallel_args = [(split_key, chunk) for chunk in chunk_files]
        pool = multiprocessing.Pool(n_jobs)
        prog_bar = tqdm(total=len(img_files), desc=f"{split_key}")

        for chunk_img_ids in pool.imap_unordered(process_images_in_chunks, parallel_args):
            img_ids[split_key] += chunk_img_ids
            prog_bar.update(len(chunk_img_ids))

            prog_bar.close()
            pool.close()

    with open(os.path.join(TARGET_PATH, "all_images.json"), "w") as fp:
        json.dump(img_ids, fp)


if __name__ == "__main__":
    argh.dispatch_command(preprocess_coco)
