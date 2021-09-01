"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolit with permission from the author.
"""

import torch
from torch import Tensor
from torch.nn import AdaptiveAvgPool3d, Conv3d, ReLU, Sigmoid
from torch.nn.modules.pooling import AdaptiveAvgPool2d

from opendr.perception.activity_recognition.x3d.algorithm.operators import Swish
from .pooling import AdaptiveAvgPoolCo3d
from .utils import unsqueezed


class SE(torch.nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
        """
        super(SE, self).__init__()
        self.avg_pool = AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = ReLU() if relu_act else Swish()
        self.fc2 = Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = Sigmoid()

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x


class ReSe(torch.nn.Module):
    """Recursive Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(
        self,
        window_size: int,
        dim_in: int,
        ratio: float,
        relu_act: bool = True,
        scope="frame",
        temporal_fill="replicate",
    ):
        """
        Args:
            window_size (int): the window size to average over
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
        """
        super(ReSe, self).__init__()
        self.avg_pool = {
            "clip": lambda: AdaptiveAvgPoolCo3d(
                window_size, output_size=(1, 1), temporal_fill=temporal_fill
            ),
            "frame": lambda: unsqueezed(AdaptiveAvgPool2d(output_size=(1, 1))),
        }[scope]()
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = unsqueezed(Conv3d(dim_in, dim_fc, 1, bias=True))
        self.fc1_act = ReLU() if relu_act else Swish()
        self.fc2 = unsqueezed(Conv3d(dim_fc, dim_in, 1, bias=True))

        self.fc2_sig = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x

    def forward3d(self, x: Tensor) -> Tensor:
        x_in = x
        for module in self.children():
            if hasattr(module, "forward3d"):
                x = module.forward3d(x)
            else:
                x = module(x)
        return x_in * x
