"""
Code from https://github.com/gmum/proto-segmentation.

NO Modifications.
"""
import os
import torch
import numpy as np

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def find_continuous_high_activation_crop(activation_map, patch_bbox, threshold, add_margin=5):
    start_h, end_h, start_w, end_w = tuple(patch_bbox)

    mask = (activation_map >= threshold).astype(int)

    stopped = [False, False, False, False]

    # greedily enlarge the high activated window
    while not all(stopped):
        if not stopped[0] and start_h > 0 and np.amax(mask[start_h-1, start_w:end_w+1]) > 0.5:
            start_h = start_h - 1
        else:
            stopped[0] = True

        if not stopped[1] and end_h < activation_map.shape[0] - 1 and np.amax(mask[end_h+1, start_w:end_w+1]) > 0.5:
            end_h = end_h + 1
        else:
            stopped[1] = True

        if not stopped[2] and start_w > 0 and np.amax(mask[start_h:end_h+1, start_w-1]) > 0.5:
            start_w = start_w - 1
        else:
            stopped[2] = True

        if not stopped[3] and end_w < activation_map.shape[1] - 1 and np.amax(mask[start_h:end_h+1, end_w+1]) > 0.5:
            end_w = end_w + 1
        else:
            stopped[3] = True

    start_h = max(start_h - add_margin, 0)
    start_w = max(start_w - add_margin, 0)
    end_h = min(end_h + add_margin, activation_map.shape[0] - 1)
    end_w = min(end_w + add_margin, activation_map.shape[1] - 1)

    return start_h, end_h+1, start_w, end_w+1
