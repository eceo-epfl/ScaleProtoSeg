"""
Preprocesss cityscapes before training a segmentation model.
https://www.cityscapes-dataset.com/

how to run run:

python -m segmentation.data.preprocess_part_cityscapes {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
import panoptic_parts as pp
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import CITYSCAPES_CATEGORIES

load_dotenv()

MARGIN_SIZE = 0

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_CITY"]
TARGET_PATH = os.environ["DATA_PATH_CITY"]
LABELS_PATH = os.path.join(SOURCE_PATH, "gtFinePanopticParts/")

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
ANNOTATIONS_DIR_PIDS = os.path.join(TARGET_PATH, "annotations_PIDS")
ANNOTATIONS_DIR_SIDS = os.path.join(TARGET_PATH, "annotations_SIDS")
ANNOTATIONS_DIR_IIDS = os.path.join(TARGET_PATH, "annotations_IIDS")


def process_images_in_chunks(args):
    split_key, city_name, png_files = args
    chunk_img_ids = []

    split_dir = os.path.join(LABELS_PATH, split_key)
    city_dir = os.path.join(split_dir, city_name)

    for file in png_files:
        img_id = file.split("_gtFinePanopticParts.tif")[0]
        chunk_img_ids.append(img_id)

        # 1. Save labels
        with open(os.path.join(city_dir, file), "rb") as f:
            label_uids = np.array(Image.open(f))
            sids, iids, pids = pp.utils.format.decode_uids(label_uids)

        np.save(os.path.join(ANNOTATIONS_DIR_PIDS, split_key, f"{img_id}.npy"), pids)
        np.save(os.path.join(ANNOTATIONS_DIR_SIDS, split_key, f"{img_id}.npy"), sids)
        np.save(os.path.join(ANNOTATIONS_DIR_IIDS, split_key, f"{img_id}.npy"), iids)

    return chunk_img_ids


def preprocess_cityscapes(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Using {len(CITYSCAPES_CATEGORIES)} object categories using {n_jobs} threads.")

    os.makedirs(ANNOTATIONS_DIR_PIDS, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR_SIDS, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR_IIDS, exist_ok=True)

    img_ids = {"train": [], "val": []}

    for split_key in tqdm(["train", "val"], desc="preprocessing images"):
        split_dir = os.path.join(LABELS_PATH, split_key)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_PIDS, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_SIDS, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_IIDS, split_key), exist_ok=True)

        for city_name in tqdm(os.listdir(split_dir), desc=split_key):
            print(f"Processing {split_key}/{city_name}...")
            city_dir = os.path.join(split_dir, city_name)
            city_files = np.asarray([file for file in os.listdir(city_dir) if file.endswith(".tif")])
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

    with open(os.path.join(TARGET_PATH, "all_images_parts.json"), "w") as fp:
        json.dump(img_ids, fp)


if __name__ == "__main__":
    argh.dispatch_command(preprocess_cityscapes)
