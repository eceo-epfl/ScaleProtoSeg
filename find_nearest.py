"""
Code from https://github.com/gmum/proto-segmentation.

NO Modifications, just one bug fix for EM, COCO, ADE
"""
import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import time

import cv2
from tqdm import tqdm
from PIL import Image

from helpers import makedir, find_high_activation_crop
from torchvision import transforms

from segmentation.constants import CITYSCAPES_MEAN, CITYSCAPES_STD
from segmentation.data.dataset import resize_label


to_normalized_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)
])


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)


class ImagePatch:

    def __init__(self, patch, label, distance,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchInfo:

    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataset,
                                         prototype_network_parallel,  # pytorch network with prototype_vectors
                                         k=5,
                                         preprocess_input_function=None,  # normalize if needed
                                         full_save=False,  # save all the images
                                         root_dir_for_saving_images='./nearest',
                                         log=print,
                                         prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()
    '''
    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    '''
    log('find nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes

    prototype_shape = prototype_network_parallel.module.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    for push_iter, img_id in tqdm(enumerate(dataset.img_ids), desc='finding nearest patches', total=len(dataset)):
        img_path = dataset.get_img_path(img_id)

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # remove margins which were used for training
        margin_size = dataset.image_margin_size
        img = img.crop((margin_size, margin_size, img.width - margin_size, img.height - margin_size))
        img_numpy = np.expand_dims(np.asarray(np.uint8(img)), 0).transpose((0, 3, 1, 2))
        img_numpy = img_numpy.astype(float) / 255.0

        with torch.no_grad():
            search_batch_input = to_normalized_tensor(img).unsqueeze(0).cuda()
            protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch_input)

        model_output_height = protoL_input_torch.shape[2]
        model_output_width = protoL_input_torch.shape[3]

        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

        search_y = np.load(os.path.join(dataset.annotations_dir, img_id + '.npy'))

        # Fix bug ADE | EM | COCO
        search_y = np.expand_dims(search_y, 0)
        if dataset.convert_targets is not None:
            search_y = dataset.convert_targets(search_y)

        # -1 because we ignore void class
        search_y = search_y - 1

        img_height = search_y.shape[1]
        img_width = search_y.shape[2]

        patch_height = img_height / model_output_height
        patch_width = img_width / model_output_width

        # protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())

        # interpolated_y = np.expand_dims(resize_label(search_y[0], size=(257, 129)).cpu().detach().numpy(), (0, 1))
        interpolated_y = np.expand_dims(resize_label(
            search_y[0], size=(proto_dist_.shape[3], proto_dist_.shape[2])
        ).cpu().detach().numpy(), (0, 1))
        # we ignore activation in 'void' class pixels
        proto_dist_ = proto_dist_ + 10e6 * (interpolated_y == -1)

        for img_idx, distance_map in enumerate(proto_dist_):
            for j in range(n_prototypes):
                target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()

                # find the closest patches in this batch to prototype j

                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])

                if full_save:
                    closest_patch_indices_in_distance_map_j = \
                        list(np.unravel_index(np.argmin(distance_map[j], axis=None),
                                              distance_map[j].shape))
                    closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                    # closest_patch_indices_in_img = \
                    # compute_rf_prototype(search_batch.size(2),
                    # closest_patch_indices_in_distance_map_j,
                    # protoL_rf_info)

                    # TODO - un-hardcode
                    # closest_patch_indices_in_img = \
                    # [
                    # 0,
                    # max(closest_patch_indices_in_distance_map_j[1] - 1, 0),
                    # min(closest_patch_indices_in_distance_map_j[1] + 1, 255),
                    # max(closest_patch_indices_in_distance_map_j[2] - 1, 0),
                    # min(closest_patch_indices_in_distance_map_j[2] + 1, 255)
                    # ]

                    closest_patch_indices_in_img = [0, 0, 0, 0, 0]

                    closest_patch_indices_in_img[1] = int(closest_patch_indices_in_distance_map_j[1] * patch_height)
                    closest_patch_indices_in_img[2] = int(
                        (closest_patch_indices_in_distance_map_j[1] + 1) * patch_height)

                    closest_patch_indices_in_img[3] = int(closest_patch_indices_in_distance_map_j[2] * patch_width)
                    closest_patch_indices_in_img[4] = int(
                        (closest_patch_indices_in_distance_map_j[2] + 1) * patch_width)

                    closest_patch = \
                        img_numpy[img_idx, :,
                        closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                        closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))

                    # ignore empty patches
                    if closest_patch.size == 0:
                        continue

                    original_img = img_numpy[img_idx]
                    original_img = np.transpose(original_img, (1, 2, 0))

                    if prototype_network_parallel.module.prototype_activation_function == 'log':
                        act_pattern = np.log(
                            (distance_map[j] + 1) / (distance_map[j] + prototype_network_parallel.module.epsilon))
                    elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                        act_pattern = max_dist - distance_map[j]
                    else:
                        act_pattern = prototype_activation_function_in_numpy(distance_map[j])

                    # 4 numbers: height_start, height_end, width_start, width_end
                    patch_indices = closest_patch_indices_in_img[1:5]

                    labels = search_y[img_idx, closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                             closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]

                    # if at least one of the pixels from the patch are from the class of the prototype,
                    # we take this as the class label
                    if np.any(labels == target_class):
                        label = target_class
                    else:
                        # in other cases, patch label = most common of classes in pixels corresponding to the patch
                        values, counts = np.unique(labels, return_counts=True)
                        label = values[np.argmax(counts)]

                    # construct the closest patch object
                    # TODO this takes lots of RAM
                    closest_patch = ImagePatch(patch=closest_patch,
                                               label=label,
                                               distance=closest_patch_distance_to_prototype_j,
                                               original_img=original_img,
                                               act_pattern=act_pattern,
                                               patch_indices=patch_indices)
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   distance=closest_patch_distance_to_prototype_j)

                # add to the j-th heap
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], closest_patch)

    # after looping through the dataset every heap will
    # have the k closest prototypes
    for j in tqdm(range(n_prototypes), desc='pruning prototypes'):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if full_save:
            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
            makedir(dir_for_saving_images)

            for i, patch in enumerate(heaps[j]):
                label = patch.label

                # save the activation pattern of the original image where the patch comes from
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i + 1) + '_act.npy'),
                        patch.act_pattern)

                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + f'_original_{label}.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)

                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                                    'nearest-' + str(i + 1) + f'_original_with_patch_{label}.png'),
                                 img_rgb=patch.original_img,
                                 bbox_height_start=patch.patch_indices[0],
                                 bbox_height_end=patch.patch_indices[1],
                                 bbox_width_start=patch.patch_indices[2],
                                 bbox_width_end=patch.patch_indices[3], color=(0, 255, 255))

                # overlay (upsampled) activation on original image and save the result
                upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                                   dsize=(patch.original_img.shape[1], patch.original_img.shape[0]),
                                                   interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + f'_original_with_heatmap_{label}.png'),
                           arr=overlayed_original_img,
                           vmin=0.0,
                           vmax=1.0)

                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                                    'nearest-' + str(i + 1) + f'_original_with_heatmap_and_patch_{label}.png'),
                                 img_rgb=overlayed_original_img,
                                 bbox_height_start=patch.patch_indices[0],
                                 bbox_height_end=patch.patch_indices[1],
                                 bbox_width_start=patch.patch_indices[2],
                                 bbox_width_end=patch.patch_indices[3], color=(0, 255, 255))

                # if different from original image, save the patch (i.e. receptive field)
                # if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                # np.save(os.path.join(dir_for_saving_images,
                # 'nearest-' + str(i+1) + '_receptive_field_indices.npy'),
                # patch.patch_indices)
                # plt.imsave(fname=os.path.join(dir_for_saving_images,
                # 'nearest-' + str(i+1) + '_receptive_field.png'),
                # arr=patch.patch,
                # vmin=0.0,
                # vmax=1.0)
                # # save the receptive field patch with heatmap
                # overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1],
                # patch.patch_indices[2]:patch.patch_indices[3], :]
                # plt.imsave(fname=os.path.join(dir_for_saving_images,
                # 'nearest-' + str(i+1) + '_receptive_field_with_heatmap.png'),
                # arr=overlayed_patch,
                # vmin=0.0,
                # vmax=1.0)

                # save the highly activated patch    
                high_act_patch_indices = find_high_activation_crop(upsampled_act_pattern)
                high_act_patch = patch.original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                 high_act_patch_indices[2]:high_act_patch_indices[3], :]
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i + 1) + f'_high_act_patch_indices_{label}.npy'),
                        high_act_patch_indices)
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + f'_high_act_patch_{label}.png'),
                           arr=high_act_patch,
                           vmin=0.0,
                           vmax=1.0)

                # save the original image with bounding box showing high activation patch
                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                                    'nearest-' + str(i + 1) + f'_high_act_patch_in_original_img_{label}.png'),
                                 img_rgb=patch.original_img,
                                 bbox_height_start=high_act_patch_indices[0],
                                 bbox_height_end=high_act_patch_indices[1],
                                 bbox_width_start=high_act_patch_indices[2],
                                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)

    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start))

    return labels_all_prototype
