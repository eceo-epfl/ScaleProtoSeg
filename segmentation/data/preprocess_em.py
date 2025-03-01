"""
Preprocess EM data.
https://imagej.net/events/isbi-2012-segmentation-challenge#training-data

how to run run:

python -m segmentation.data.preprocess_em {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import EM_RGB_2_ID, EM_VAL_SIZE
from segmentation.utils import add_margins_to_image

load_dotenv()

MARGIN_SIZE = 0

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_EM"]
TARGET_PATH = os.environ["DATA_PATH_EM"]
LABELS_PATH = os.path.join(SOURCE_PATH, "train-labels.tif")
IMAGES_PATH = os.path.join(SOURCE_PATH, "train-volume.tif")

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
MARGIN_IMG_DIR = os.path.join(TARGET_PATH, f"img_with_margin_{MARGIN_SIZE}")
id_2_train_id_vec = np.vectorize(EM_RGB_2_ID.get)


def process_images_in_chunks(args):
    split_key, img_ids = args
    chunk_img_ids = []
    images = Image.open(IMAGES_PATH)
    labels = Image.open(LABELS_PATH)

    for img_id in img_ids:
        chunk_img_ids.append(str(img_id))
        labels.seek(img_id)

        # 1. Save labels
        label_ids = np.array(labels.convert("L"))
        label_ids = id_2_train_id_vec(label_ids).astype(np.uint8)
        np.save(os.path.join(ANNOTATIONS_DIR, split_key, f"{img_id}.npy"), label_ids)

        # 2. Save image (with mirrored margin)
        images.seek(img_id)
        img = images.convert("RGB")
        img_with_margin = add_margins_to_image(img, MARGIN_SIZE)
        output_img_path = os.path.join(MARGIN_IMG_DIR, split_key, str(img_id) + ".png")
        img_with_margin.save(output_img_path)

    return chunk_img_ids


def preprocess_em(n_jobs: int, chunk_size: int = 1, seed: int = 42):
    n_jobs = int(n_jobs)
    print(f"Using {len(EM_RGB_2_ID)} object categories using {n_jobs} threads.")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(MARGIN_IMG_DIR, exist_ok=True)

    img_ids = {"train": [], "val": []}

    images = Image.open(IMAGES_PATH)
    np.random.seed(seed)
    val_ids = np.random.choice(images.n_frames, EM_VAL_SIZE, replace=False).tolist()
    train_ids = [i for i in range(images.n_frames) if i not in val_ids]
    mapping_ids = {"train": train_ids, "val": val_ids}

    for split_key in tqdm(["train", "val"], desc="preprocessing images"):
        os.makedirs(os.path.join(MARGIN_IMG_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)

        print(f"Processing {split_key}")
        img_files = mapping_ids[split_key]
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
    argh.dispatch_command(preprocess_em)
