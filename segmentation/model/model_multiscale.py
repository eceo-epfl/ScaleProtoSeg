"""Define the pytorch model for multiscale-prototype learning: ScaleProtoSeg prototype phase"""

from typing import Any, List, Optional, Tuple, Union

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from receptive_field import compute_proto_layer_rf_info_v2
from segmentation.model.deeplab_features import (
    deeplabv2_resnet101_features,
    deeplabv2_resnet101_features_multiscale,
    deeplabv2_vgg16_features_multiscale,
    deeplabv3_resnet101_features_multiscale,
    unet_aspp_features,
)
from segmentation.model.densenet_features import (
    densenet121_features,
    densenet161_features,
    densenet169_features,
    densenet201_features,
)
from segmentation.model.resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet101_features,
    resnet152_features,
)
from segmentation.model.scale_head import WeightedAgg
from segmentation.model.vgg_features import (
    vgg11_bn_features,
    vgg11_features,
    vgg13_bn_features,
    vgg13_features,
    vgg16_bn_features,
    vgg16_features,
    vgg19_bn_features,
    vgg19_features,
)
from settings import log

base_architecture_to_features = {
    "resnet18": resnet18_features,
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "densenet121": densenet121_features,
    "densenet161": densenet161_features,
    "densenet169": densenet169_features,
    "densenet201": densenet201_features,
    "deeplabv2_resnet101": deeplabv2_resnet101_features,
    "deeplabv2_resnet101_multiscale": deeplabv2_resnet101_features_multiscale,
    "deeplabv3_resnet101_multiscale": deeplabv3_resnet101_features_multiscale,
    "deeplabv2_vgg16_multiscale": deeplabv2_vgg16_features_multiscale,
    "unet_aspp": unet_aspp_features,
    "vgg11": vgg11_features,
    "vgg11_bn": vgg11_bn_features,
    "vgg13": vgg13_features,
    "vgg13_bn": vgg13_bn_features,
    "vgg16": vgg16_features,
    "vgg16_bn": vgg16_bn_features,
    "vgg19": vgg19_features,
    "vgg19_bn": vgg19_bn_features,
}


@gin.configurable(allowlist=["bottleneck_stride", "patch_classification", "num_scales"])
class PPNetMultiScale(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        img_size: int,
        prototype_shape: Tuple[int, int, int, int],
        proto_layer_rf_info: List[float],
        num_classes: int,
        init_weights: bool = True,
        prototype_activation_function: str = "log",
        add_on_layers_type: str = "bottleneck",
        bottleneck_stride: Optional[int] = None,
        patch_classification: bool = False,
        num_scales: int = 4,
        scale_head_type: Optional[str] = None,
    ):
        """Initialize the PPNetMultiScale

        Args:
            features (nn.Module): Segmentation architecture to compute the features for the prototype layer.
            img_size (int): Size of the input image.
            prototype_shape (Tuple[int, int, int, int]): Latent dimensions of the prototype layer.
            proto_layer_rf_info (List[float]): Information on the prototype layer (NOT used).
            num_classes (int): Number of classes for the segmentation class.
            init_weights (bool): Flag for the initialization of the model weights.
            prototype_activation_function (str): Type of activation function for the prototype layer. Defaults to "log".
            add_on_layers_type (str): Type of add-on layer between segmentation features and prototype layer. Defaults to "bottleneck.
            bottleneck_stride (int, optional): Stride for the bottleneck add-on. Defaults to None.
            patch_classification (bool): Flag for semantic segmentation. Defaults to False.
            num_scales (int): Number of scales in the features outputs and prototype layer. Defaults to 4.
            scale_head_type (str, optional): Aggregation layer type to propagate previous scale information. Defaults to None.
        """

        super(PPNetMultiScale, self).__init__()
        self.img_size = img_size
        self.epsilon = 1e-4
        self.bottleneck_stride = bottleneck_stride
        self.patch_classification = patch_classification
        self.num_scales = num_scales

        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)

        # Option to accumulate information across scale in the prototype layer
        if scale_head_type is not None:
            self.scale_head = WeightedAgg(output_type=scale_head_type, channel_dim=prototype_shape[1])
        else:
            self.scale_head = None

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        """
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        """
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)

        # Initialization of prototype to class mapping across scales.
        num_prototypes_per_scale = self.num_prototypes // self.num_scales
        num_prototypes_per_class_scale = self.num_prototypes // self.num_classes // self.num_scales
        for i in range(self.num_scales):
            for j in range(self.num_classes):
                self.prototype_class_identity[
                    i * num_prototypes_per_scale
                    + j * num_prototypes_per_class_scale : i * num_prototypes_per_scale
                    + (j + 1) * num_prototypes_per_class_scale,
                    j,
                ] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # This mapping linked prototype Ids to scale.
        self.scale_num_prototypes = {
            scale: (scale * num_prototypes_per_scale, (scale + 1) * num_prototypes_per_scale)
            for scale in range(self.num_scales)
        }

        # Set-up features and information on output channel dimensions.
        self.features = features
        features_name = str(self.features).upper()
        if features_name.startswith("VGG") or features_name.startswith("RES"):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][
                -1
            ].out_channels
        elif features_name.startswith("DENSE"):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][
                -1
            ].num_features
        elif features_name.startswith("DEEPLAB"):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][
                -2
            ].out_channels
        elif features_name.startswith("MSC"):
            first_add_on_layer_in_channels = [i for i in features.base.modules() if isinstance(i, nn.Conv2d)][
                -2
            ].out_channels
        else:
            raise Exception(f"{features_name[:10]} base_architecture NOT implemented")

        # Set-up the add-on layers before prototype layer.
        add_on_layers = []
        if add_on_layers_type == "bottleneck_pool":
            add_on_layers.append(
                nn.Conv2d(
                    in_channels=first_add_on_layer_in_channels,
                    out_channels=first_add_on_layer_in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=self.bottleneck_stride,
                )
            )
            add_on_layers.append(nn.ReLU())

        if add_on_layers_type.startswith("bottleneck"):
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(
                    nn.Conv2d(in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1)
                )
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(
                    nn.Conv2d(in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1)
                )
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert current_out_channels == self.prototype_shape[1]
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)

        elif add_on_layers_type == "deeplab_simple":
            log("deeplab_simple add_on_layers")
            self.add_on_layers = nn.Sequential(nn.Sigmoid())

        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1
                ),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            )

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # Last classification layer from prototypes to classes.
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        # Initialization of the weights.
        if init_weights:
            self._initialize_weights()

    @property
    def prototype_shape(self) -> Tuple[int, int, int, int]:
        return self.prototype_vectors.shape

    @property
    def num_prototypes(self) -> int:
        return self.prototype_vectors.shape[0]

    @property
    def num_classes(self) -> int:
        return self.prototype_class_identity.shape[1]

    def run_last_layer(self, prototype_activations: torch.Tensor) -> torch.Tensor:
        return self.last_layer(prototype_activations)

    def conv_features(self, x: torch.Tensor):
        x = self.features(x)
        # multi-scale training (MCS)
        if isinstance(x, list):
            return [self.add_on_layers(x_scaled) for x_scaled in x]

        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _l2_convolution(x: torch.Tensor, prototype_vectors: torch.Tensor, ones: torch.Tensor) -> torch.Tensor:
        """Apply self.prototype_vectors as l2-convolution filters on input x

        Args:
            x (torch.Tensor): Output from the last feature maps in the segmentation architecture.
            prototype_vectors (torch.Tensor): Vectors in the latent space that represent the vectors.
            ones (torch.Tensor): Weights for the convolution.

        Returns:
            (torch.Tensor): Output distance between x and prototype vectors
        """
        x2 = x**2
        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototype_vectors**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _scale_l2_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the l2 convolution at all existing scales.

        Args:
            x (torch.Tensor): Output from the last feature maps in the segmentation architecture.

        Returns:
            (torch.Tensor): Output distance between x and prototype vectors
        """

        # Reshape the last feature map and prototype vectors to extract the scale dimension.
        B, C, H, W = x.shape
        x = x.view(B, self.num_scales, int(C / self.num_scales), H, W)
        proto = self.prototype_vectors.view(self.num_prototypes, int(C / self.num_scales), 1, 1)
        ones = self.ones.view(self.num_prototypes, int(C / self.num_scales), 1, 1)

        # Loop through scale to compute distances
        out_list = []
        for i in range(self.num_scales - 1, -1, -1):
            x_scale = torch.squeeze(x[:, i], dim=1)
            proto_scale = proto[self.scale_num_prototypes[i][0] : self.scale_num_prototypes[i][1]]
            ones_scale = ones[self.scale_num_prototypes[i][0] : self.scale_num_prototypes[i][1]]

            if self.scale_head is not None and i < self.num_scales - 1:
                activations = self.distance_2_similarity(out_list[-1])
                x_scale = self.scale_head(
                    x_scale,
                    activations=activations,
                    prototypes=torch.squeeze(
                        proto[self.scale_num_prototypes[i + 1][0] : self.scale_num_prototypes[i + 1][1]], dim=0
                    ),
                )
            out_list.append(self._l2_convolution(x_scale, proto_scale, ones_scale))

        return torch.concat(out_list[::-1], dim=1)

    def prototype_distances(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_features(x)
        distances = self._scale_l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        conv_features = self.conv_features(x)
        # MCS
        if isinstance(conv_features, list):
            return [self.forward_from_conv_features(c, **kwargs) for c in conv_features]

        return self.forward_from_conv_features(conv_features, **kwargs)

    def forward_from_conv_features(
        self,
        conv_features: Union[torch.Tensor, List[torch.Tensor]],
        return_activations: bool = False,
        return_distances: bool = False,
    ) -> Any:
        """Compute the forward pass from the feature maps of the segmentation architecture

        Args:
            conv_features (Union[torch.Tensor, List[torch.Tensor]]): Feature maps.
            return_activations (bool, optional): Flag to return prototype activations. Defaults to False.
            return_distances (bool, optional): Flag to return distances to prototypes. Defaults to False.


        Returns:
            Any: Model Outputs
        """

        if isinstance(conv_features, list):
            return [self.forward_from_conv_features(c) for c in conv_features]

        # distances.shape = (batch_size, num_prototypes, n_patches_cols, n_patches_rows)
        distances = self._scale_l2_convolution(conv_features)

        if hasattr(self, "patch_classification") and self.patch_classification:
            # flatten to get predictions per patch
            batch_size, num_prototypes, n_patches_cols, n_patches_rows = distances.shape

            # shape: (batch_size, n_patches_cols, n_patches_rows, num_prototypes)
            dist_view = distances.permute(0, 2, 3, 1).contiguous()
            dist_view = dist_view.reshape(-1, num_prototypes)
            prototype_activations = self.distance_2_similarity(dist_view)

            logits = self.run_last_layer(prototype_activations)

            # shape: (batch_size, n_patches_cols, n_patches_rows, num_classes)
            logits = logits.reshape(batch_size, n_patches_cols, n_patches_rows, -1)

            if return_activations and not return_distances:
                return logits, prototype_activations

            elif return_activations and return_distances:
                return logits, distances, prototype_activations

            else:
                return logits, distances

        else:
            raise Exception("Original Prototype Network Implementation")

    def push_forward(self, x: torch.Tensor) -> Any:
        """this method is needed for the pushing operation"""
        conv_output = self.conv_features(x)

        if isinstance(conv_output, list):
            return [(c, self._scale_l2_convolution(c)) for c in conv_output]

        distances = self._scale_l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune: List[int]):
        """Drops prototypes from model based on input list of indices.

        Args:
            prototypes_to_prune (List[int]): List of indices that indicates the prototypes to be removed
        """
        prototypes_to_keep = list(sorted(set(range(self.num_prototypes)) - set(prototypes_to_prune)))

        for scale in range(self.num_scales):

            prototypes_to_keep_scale = list(
                sorted(
                    set(range(self.scale_num_prototypes[scale][0], self.scale_num_prototypes[scale][1]))
                    - set(prototypes_to_prune)
                )
            )

            if scale == 0:
                self.scale_num_prototypes[scale] = (0, len(prototypes_to_keep_scale))
            else:
                self.scale_num_prototypes[scale] = (
                    self.scale_num_prototypes[scale - 1][1],
                    self.scale_num_prototypes[scale - 1][1] + len(prototypes_to_keep_scale),
                )

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True)

        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        rep = (
            "PPNet(\n"
            "\tfeatures: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )
        return rep.format(
            self.features, self.img_size, self.prototype_shape, self.proto_layer_rf_info, self.num_classes, self.epsilon
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength: float):
        """Set the frozen weights for the last layer after the prototypes

        Args:
            incorrect_strength (float): Negative strength used when the prototypes is not assigned to the class.
        """
        incorrect_class_connection = incorrect_strength
        correct_class_connection = 1

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


@gin.configurable(denylist=["img_size"])
def construct_PPNet(
    img_size: int = 224,
    base_architecture: nn.Module = gin.REQUIRED,
    pretrained: bool = True,
    prototype_shape: Tuple[int, int, int, int] = (2000, 512, 1, 1),
    num_classes: int = 200,
    prototype_activation_function: str = "log",
    add_on_layers_type: str = "bottleneck",
    scale_head_type: Optional[str] = None,
) -> PPNetMultiScale:
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    if hasattr(features, "conv_info"):
        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    else:
        layer_filter_sizes, layer_strides, layer_paddings = [], [], []

    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )

    return PPNetMultiScale(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
        scale_head_type=scale_head_type,
    )
