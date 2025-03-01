"""
Preprocesss PASCAL Panoptic Parta dataset before quantitative interpretable evaluation.
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

how to run run:

python -m segmentation.data.preprocess_part_pascal {N_JOBS}
"""

import json
import multiprocessing
import os

import argh
import numpy as np
import panoptic_parts as pp
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from segmentation.constants import MAX_PARTS_PASCAL

load_dotenv()

SOURCE_PATH = os.environ["SOURCE_DATA_PATH_PASCAL"]
TARGET_PATH = os.environ["DATA_PATH_PASCAL"]

ANNOTATIONS_DIR = os.path.join(TARGET_PATH, "annotations")
ANNOTATIONS_DIR_PIDS = os.path.join(TARGET_PATH, "annotations_PIDS_PRE")
ANNOTATIONS_DIR_PIDS_CLEAN = os.path.join(TARGET_PATH, "annotations_PIDS")
ANNOTATIONS_DIR_SIDS = os.path.join(TARGET_PATH, "annotations_SIDS")
ANNOTATIONS_DIR_IIDS = os.path.join(TARGET_PATH, "annotations_IIDS")


def prep_mapping():
    MAPPING_FILE = os.path.join(SOURCE_PATH, "parts.yaml")

    with open(MAPPING_FILE, "r") as f:
        mapping = yaml.safe_load(f)
        class2part = mapping["scene_class2part_classes"]
        class_parts_pids = mapping["countable_pids_groupings"]

    class2id = {k: i + 1 for i, k in enumerate(class2part.keys())}

    # Create a mapping from class to part to pid
    class_part_pids_real = {}
    for key, value in class2part.items():
        class_part_pids_real[key] = {part: index + 1 for index, part in enumerate(value)}

    # Create a mapping from class to part to pid | Should cover 0 and -1 cases
    final_mapping = {0: {-1: -1, 0: 0}}
    for key, value in class_part_pids_real.items():
        part_mapping = {-1: -1, 0: 0}
        for name, pid in value.items():
            if key in class_parts_pids and name in class_parts_pids[key]:
                for f_pid in class_parts_pids[key][name]:
                    part_mapping[f_pid] = pid
            else:
                part_mapping[pid] = pid

        final_mapping[class2id[key]] = part_mapping

    return final_mapping


def process_images_in_chunks(args):
    split_key, img_ids, final_mapping = args
    chunk_img_ids = []

    unique_classes = set()

    def apply_mapping(sid, pid):
        return final_mapping[sid].get(pid)

    vectorized_apply_mapping = np.vectorize(apply_mapping)

    for img_id in img_ids:
        chunk_img_ids.append(img_id)

        # 1. Save labels
        if split_key != "test":
            with open(os.path.join(SOURCE_PATH, "labels/", split_key, f"{img_id}.tif"), "rb") as f:
                label_uids = np.array(Image.open(f))
                sids, iids, pids = pp.utils.format.decode_uids(label_uids)

            pids_clean = vectorized_apply_mapping(sids, pids)

            if np.max(pids_clean) > MAX_PARTS_PASCAL:
                raise ValueError("Error in Post-Processing")

            np.save(os.path.join(ANNOTATIONS_DIR_PIDS, split_key, f"{img_id}.npy"), pids)
            np.save(os.path.join(ANNOTATIONS_DIR_PIDS_CLEAN, split_key, f"{img_id}.npy"), pids_clean)
            np.save(os.path.join(ANNOTATIONS_DIR_SIDS, split_key, f"{img_id}.npy"), sids)
            np.save(os.path.join(ANNOTATIONS_DIR_IIDS, split_key, f"{img_id}.npy"), iids)

    return chunk_img_ids, unique_classes


def preprocess_pascal(n_jobs: int, chunk_size: int = 10):
    n_jobs = int(n_jobs)
    print(f"Preprocessing PASCAL VOC 2012")

    os.makedirs(ANNOTATIONS_DIR_PIDS, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR_SIDS, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR_IIDS, exist_ok=True)

    img_ids = {"train": [], "val": []}

    for split_key in tqdm(["train", "val"], desc="preprocessing images"):
        split_img_ids = [
            img_id.strip().split("/")[-1].split(".")[0]
            for img_id in os.listdir(os.path.join(SOURCE_PATH, f"labels/{split_key}"))
        ]

        os.makedirs(os.path.join(ANNOTATIONS_DIR, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_PIDS, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_PIDS_CLEAN, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_SIDS, split_key), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATIONS_DIR_IIDS, split_key), exist_ok=True)

        n_chunks = int(np.ceil(len(split_img_ids) / chunk_size))
        chunk_files = np.array_split(split_img_ids, n_chunks)
        final_mapping = prep_mapping()
        parallel_args = [(split_key, chunk, final_mapping) for chunk in chunk_files]

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

    with open(os.path.join(TARGET_PATH, "all_images_parts.json"), "w") as fp:
        json.dump(img_ids, fp)


if __name__ == "__main__":
    argh.dispatch_command(preprocess_pascal)
