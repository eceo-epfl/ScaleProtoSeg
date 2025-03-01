"""
Code from https://github.com/gmum/proto-segmentation.

Extended to other datasets.
"""

import os

from log import create_logger
from dotenv import load_dotenv

load_dotenv()

base_architecture = 'vgg19'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

data_path = {"cityscapes": os.environ['DATA_PATH_CITY'],
             "pascal": os.environ['DATA_PATH_PASCAL'],
             "ade": os.environ['DATA_PATH_ADE'],
             "em": os.environ['DATA_PATH_EM'],
             "coco": os.environ['DATA_PATH_COCO']}
# train_dir = os.environ['TRAIN_DIR'] TODO: Define these
# test_dir = os.environ['TEST_DIR']
# train_push_dir = os.environ['TRAIN_PUSH_DIR']
log_dir = os.environ['LOG_DIR']

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

os.makedirs(log_dir, exist_ok=True)
log, logclose = create_logger(log_filename=os.path.join(log_dir, 'logger.log'))
