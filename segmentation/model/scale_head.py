"""Code for extending current scale feature maps with previous scale information based on prototye activations"""

import torch
import torch.nn as nn


class WeightedAgg(nn.Module):
    """Layer to aggregate previous scale activations with current feature map: x"""

    def __init__(self, output_type: str, **kwargs) -> None:
        super().__init__()

        self.output_layer = factory_output_layer(output_type, **kwargs)

    def forward(self, x: torch.Tensor, activations: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:

        weight_sum = prototypes[None, :, :, :, :] * activations[:, :, None, :, :]
        weight_sum = weight_sum.sum(dim=1)

        output = self.output_layer(x, weight_sum)

        return output


"""Different aggregation strategies"""


class OutSum(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        return (x + scale_x) / 2


class OutMult(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * scale_x)


class OutConcat(nn.Module):
    def __init__(self, channel_dim: int) -> None:
        super().__init__()
        self.linear_block = nn.Sequential(nn.Linear(2 * channel_dim, channel_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        x_tot = torch.cat((x, scale_x), dim=1)
        x_tot = x_tot.permute(0, 2, 3, 1)
        x_tot = self.linear_block(x_tot)
        return x_tot.permute(0, 3, 1, 2)


def factory_output_layer(output_type: str, **kwargs) -> nn.Module:
    """Function factory to access the different strategies"""
    if output_type == "sum":
        return OutSum()
    elif output_type == "mult":
        return OutMult()
    elif output_type == "concat":
        return OutConcat(**kwargs)
    else:
        raise NotImplementedError(f"Unknown output type: {output_type}")
