from typing import Optional

import gin

from deeplab_pytorch.libs.models.deeplabv2 import DeepLabV2, DeepLabV2_VGG
from deeplab_pytorch.libs.models.deeplabv2_multiscale import DeepLabV2 as DeepLabV2Scale
from deeplab_pytorch.libs.models.deeplabv2_multiscale import (
    DeepLabV2_VGG as DeepLabV2_VGGScale,
)
from deeplab_pytorch.libs.models.deeplabv2_multiscaleplus import (
    DeepLabV2 as DeepLabV2ScalePlus,
)
from deeplab_pytorch.libs.models.deeplabv3_multiscale import DeepLabV3 as DeepLabV3Scale
from deeplab_pytorch.libs.models.unet import UNet, UNetASPP
from segmentation.utils import MSC


def torchvision_resnet_weight_key_to_deeplab2(key: str) -> Optional[str]:
    """Matched each key from pre-trained ResNet to DeepLabv2"""
    segments = key.split(".")

    if segments[0].startswith("layer"):
        layer_num = int(segments[0].split("layer")[-1])
        dl_layer_num = layer_num + 1

        block_num = int(segments[1])
        dl_block_str = f"block{block_num + 1}"

        layer_type = segments[2]
        if layer_type == "downsample":
            shortcut_module_num = int(segments[3])
            if shortcut_module_num == 0:
                module_name = "conv"
            elif shortcut_module_num == 1:
                module_name = "bn"
            else:
                raise ValueError(shortcut_module_num)

            return f"layer{dl_layer_num}.{dl_block_str}.shortcut.{module_name}.{segments[-1]}"

        else:
            layer_type, conv_num = segments[2][:-1], segments[2][-1]
            conv_num = int(conv_num)

            if conv_num == 1:
                dl_conv_name = "reduce"
            elif conv_num == 2:
                dl_conv_name = "conv3x3"
            elif conv_num == 3:
                dl_conv_name = "increase"
            else:
                raise ValueError(conv_num)

            return f"layer{dl_layer_num}.{dl_block_str}.{dl_conv_name}.{layer_type}.{segments[-1]}"

    elif segments[0] in {"conv1", "bn1"}:
        layer_type = segments[0][:-1]
        return f"layer1.conv1.{layer_type}.{segments[-1]}"

    return None


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_resnet101_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs):
    return MSC(
        base=DeepLabV2(n_classes=deeplab_n_features, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_resnet50_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs):
    return MSC(
        base=DeepLabV2(n_classes=deeplab_n_features, n_blocks=[3, 4, 6, 3], atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_vgg16_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs):
    return MSC(
        base=DeepLabV2_VGG(n_classes=deeplab_n_features, atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["unet_n_features", "scales"])
def unet_features(pretrained=False, unet_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs):
    return MSC(
        base=UNet(
            n_channels=3,
            n_classes=unet_n_features,
        ),
        scales=scales,
    )


@gin.configurable(allowlist=["unet_n_features", "multiscale", "scales"])
def unet_aspp_features(pretrained=False, unet_n_features: int = gin.REQUIRED, multiscale=False, scales=[1.0], **kwargs):
    return MSC(
        base=UNetASPP(
            n_channels=3,
            out_features=unet_n_features,
            n_classes=unet_n_features,
            rates=[6, 12, 18, 24],
            multiscale=multiscale,
        ),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_resnet101_features_multiscale(
    pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs
):
    return MSC(
        base=DeepLabV2Scale(n_classes=deeplab_n_features, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_resnet50_features_multiscale(
    pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs
):
    return MSC(
        base=DeepLabV2Scale(n_classes=deeplab_n_features, n_blocks=[3, 4, 6, 3], atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_vgg16_features_multiscale(
    pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs
):
    return MSC(
        base=DeepLabV2_VGGScale(n_classes=deeplab_n_features, atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv3_resnet101_features_multiscale(
    pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs
):
    return MSC(
        base=DeepLabV3Scale(
            n_classes=deeplab_n_features,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        ),
        scales=scales,
    )


@gin.configurable(allowlist=["deeplab_n_features", "scales"])
def deeplabv2_resnet101_features_multiscaleplus(
    pretrained=False, deeplab_n_features: int = gin.REQUIRED, scales=[1.0], **kwargs
):
    return MSC(
        base=DeepLabV2ScalePlus(n_classes=deeplab_n_features, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]),
        scales=scales,
    )
