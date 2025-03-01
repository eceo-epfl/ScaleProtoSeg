"""
Preprocess ADE20k data.
http://sceneparsing.csail.mit.edu/

how to run run:

python -m segmentation.data.preprocess_ade {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import ADE20k_ID_2_LABEL
from segmentation.utils import add_margins_to_image

load_dotenv()

MARGIN_SIZE = 0

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_ADE"]
TARGET_PATH = os.environ["DATA_PATH_ADE"]
LABELS_PATH = os.path.join(SOURCE_PATH, "annotations/")

# Need to rename the raw annotation folder
os.rename(LABELS_PATH, os.path.join(SOURCE_PATH, "annotations_png/"))
LABELS_PATH = os.path.join(SOURCE_PATH, "annotations_png/")

IMAGES_PATH = os.path.join(SOURCE_PATH, "images/")

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, f"img_with_margin_{MARGIN_SIZE}")


def process_images_in_chunks(args):
    split_key, split_key_input, png_files = args
    chunk_img_ids = []

    split_dir = os.path.join(LABELS_PATH, split_key_input)

    for file in png_files:
        img_id = file.split(".png")[0]
        chunk_img_ids.append(img_id)

        # 1. Save labels
        with open(os.path.join(split_dir, file), "rb") as f:
            label_ids = np.array(Image.open(f).convert("RGB"))[:, :, 0]
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, f"{img_id}.npy"), label_ids)

        # 2. Save image (with mirrored margin)
        input_img_path = os.path.join(IMAGES_PATH, split_key_input, img_id + ".jpg")
        with open(input_img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img_with_margin = add_margins_to_image(img, MARGIN_SIZE)
        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, img_id + ".png")
        img_with_margin.save(output_img_path)

    return chunk_img_ids


def preprocess_ade(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Using {len(ADE20k_ID_2_LABEL)} object categories using {n_jobs} threads.")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {"train": [], "val": []}

    for split_key, split_key_input in tqdm(
        zip(["train", "val"], ["training", "validation"]), desc="preprocessing images"
    ):
        split_dir = os.path.join(LABELS_PATH, split_key_input)
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        print(f"Processing {split_key}")
        img_files = np.asarray([file for file in os.listdir(split_dir) if file.endswith(".png")])
        n_chunks = int(np.ceil(len(img_files) / chunk_size))
        chunk_files = np.array_split(img_files, n_chunks)

        parallel_args = [(split_key, split_key_input, chunk) for chunk in chunk_files]
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
    argh.dispatch_command(preprocess_ade)
