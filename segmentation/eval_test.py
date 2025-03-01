"""
Original code for the test inference from https://github.com/gmum/proto-segmentation.

Marginal Modifications.
"""
import os

import argh
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, PASCAL_ID_MAPPING, CITYSCAPES_CATEGORIES, \
    CITYSCAPES_ID_2_LABEL
from settings import data_path, log


def run_evaluation(model_name: str, training_phase: str, batch_size: int, pascal: str,
                   margin: int = 0):

    batch_size = int(batch_size)
    pascal = (pascal == "True")
    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    dataset_path = data_path["pascal"] if pascal else data_path["cityscapes"]

    if training_phase == 'pruned':
        checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
    else:
        checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    log(f'Loading model from {checkpoint_path}')
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    img_dir = os.path.join(dataset_path, f'img_with_margin_{margin}/test')

    all_img_files = [p for p in os.listdir(img_dir) if p.endswith('.npy')]


    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES

    OUR_ID_2_SOURCE_ID = {v: k for k, v in ID_MAPPING.items()}
    if not pascal:
        OUR_ID_2_SOURCE_ID[0] = 0

        rev_origin = {v: k for k, v in CITYSCAPES_ID_2_LABEL.items()}

        OUR_ID_2_SOURCE_ID = {k: rev_origin[CITYSCAPES_CATEGORIES[v]] for k, v in OUR_ID_2_SOURCE_ID.items()}
    OUR_ID_2_SOURCE_ID = np.vectorize(OUR_ID_2_SOURCE_ID.get)

    RESULTS_DIR = os.path.join(model_path, f'evaluation/test/{training_phase}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.random.shuffle(all_img_files)

    n_batches = int(np.ceil(len(all_img_files) / batch_size))
    batched_img_files = np.array_split(all_img_files, n_batches)

    with torch.no_grad():
        for batch_img_files in tqdm(batched_img_files, desc='evaluating'):
            img_tensors = []
            img_arrays = []

            for img_file in batch_img_files:
                img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)

                if margin != 0:
                    img = img[margin:-margin, margin:-margin]

                img_arrays.append(img)

                if pascal:
                    img_shape = (513, 513)
                else:
                    img_shape = img.shape

                img_tensor = transform(img)
                if pascal:
                    img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0),
                                                                 size=img_shape, mode='bilinear', align_corners=False)[0]
                img_tensors.append(img_tensor)

            img_tensors = torch.stack(img_tensors, dim=0).cuda()
            batch_logits, batch_distances = ppnet.forward(img_tensors)
            del batch_distances, img_tensor

            batch_logits = batch_logits.permute(0, 3, 1, 2)

            for sample_i in range(len(batch_img_files)):
                img = img_arrays[sample_i]
                logits = torch.unsqueeze(batch_logits[sample_i], 0)

                logits = F.interpolate(logits, size=(img.shape[0], img.shape[1]), mode='bilinear', align_corners=False)[0]
                pred = torch.argmax(logits, dim=0).cpu().detach().numpy()

                pred = pred + 1
                pred = OUR_ID_2_SOURCE_ID(pred)

                pred_img = Image.fromarray(np.uint8(pred))

                img_id = batch_img_files[sample_i].split('.')[0]
                pred_img.convert("L").save(os.path.join(RESULTS_DIR, f'{img_id}.png'))


if __name__ == '__main__':
    argh.dispatch_command(run_evaluation)
