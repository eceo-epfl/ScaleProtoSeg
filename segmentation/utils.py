"""
Utility functions for segmentation models. Code from https://github.com/gmum/proto-segmentation.

Extension for multi-scale use case.
"""

from PIL import Image
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import log


def add_margins_to_image(img, margin_size):
    margin_left = img.crop((0, 0, margin_size, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_right = img.crop((img.width - margin_size, 0, img.width, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_top = img.crop((0, 0, img.width, margin_size)).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom = img.crop((0, img.height - margin_size, img.width, img.height)).transpose(Image.FLIP_TOP_BOTTOM)

    margin_top_left = img.crop((0, 0, margin_size, margin_size)).transpose(Image.FLIP_LEFT_RIGHT).transpose(
        Image.FLIP_TOP_BOTTOM)
    margin_top_right = img.crop((img.width - margin_size, 0, img.width, margin_size)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_left = img.crop((0, img.height - margin_size, margin_size, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_right = img.crop(
        (img.width - margin_size, img.height - margin_size, img.width, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    concat_img = Image.new('RGB', (img.width + margin_size * 2, img.height + margin_size * 2))

    concat_img.paste(img, (margin_size, margin_size))
    concat_img.paste(margin_left, (0, margin_size))
    concat_img.paste(margin_right, (img.width + margin_size, margin_size))
    concat_img.paste(margin_top, (margin_size, 0))
    concat_img.paste(margin_bottom, (margin_size, img.height + margin_size))
    concat_img.paste(margin_top_left, (0, 0))
    concat_img.paste(margin_top_right, (img.width + margin_size, 0))
    concat_img.paste(margin_bottom_left, (0, img.height + margin_size))
    concat_img.paste(margin_bottom_right, (img.width + margin_size, img.height + margin_size))

    return concat_img


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                print(m[0])
                if isinstance(m[1], nn.Conv2d):
                    print("In", m[0])
                    yield m[1].bias


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales is not None:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)

        if len(self.scales) == 0:
            return logits
        elif len(self.scales) > 0 and isinstance(logits, list):
            raise NotImplementedError("MSC and multiscale outptus are not supported yet.") # TODO: implement this

        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max

@torch.no_grad()
def projection_simplex_sort(v, z=1):
    """Project v to the simplex"""
    n_features = v.size(1)
    u,_ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u,1) - z
    ind = torch.arange(n_features).type_as(v) + 1
    cond = u - cssv / ind > 0
    rho,ind_rho = (ind*cond).max(1)
    theta = torch.gather(cssv,1,ind_rho[:,None]) / rho[:,None]
    w = torch.clamp(v - theta, min=0)
    return w


def freezing_batch_norm(module: LightningModule):
    """Freeze the BatchNorm layers"""
    if module.freeze_type == 'all':
        module.ppnet.features.base.freeze_bn()
    elif module.freeze_type == 'aspp_bn':
        module.ppnet.features.base.freeze_bn(freeze_aspp_bn=False)
    elif module.freeze_type == 'none':
        pass
    else:
        raise ValueError(f'Unknown freeze type: {module.freeze_type}')



def non_zero_proto(model):
    """Extract the used prototypes by grouping mechanism"""

    set_proto_idx = []

    for cls_i in range(len(model.group_projection)):
        list_proto_idx = torch.nonzero(model.prototype_class_identity[:, cls_i]).flatten().tolist() # Approximation CITYSCAPES, ADE, PASCAL

        for k in range(model.num_groups):
            group_weight = model.group_projection[cls_i].weight.data[k, :]
            non_zero_proto_id = torch.nonzero(group_weight).flatten().tolist()
            real_proto_id = [list_proto_idx[group_proto_id] for group_proto_id in non_zero_proto_id]
            set_proto_idx.extend(real_proto_id)

    return np.unique(np.array(set_proto_idx)).tolist()

