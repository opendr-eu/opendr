from collections import OrderedDict

import continual as co
import torch
from torch import Tensor, nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from opendr.perception.activity_recognition.x3d.algorithm.operators import Swish


def _round_width(width, multiplier, min_width=8, divisor=8):
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


class SE(torch.nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = _round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)
        self.fc2_sig = nn.Sigmoid()
        self.dim_in = dim_in
        self.ratio = ratio
        self.relu_act = relu_act

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x


def CoSe(
    window_size: int,
    dim_in: int,
    ratio: float,
    relu_act: bool = True,
    scope="frame",
    temporal_fill="zeros",
):
    dim_fc = _round_width(dim_in, ratio)
    return co.Residual(
        co.Sequential(
            OrderedDict(
                [
                    (
                        "avg_pool",
                        co.AdaptiveAvgPool3d(
                            output_size=(1, 1, 1),
                            kernel_size={
                                "clip": window_size,
                                "frame": 1,
                            }[scope],
                            temporal_fill=temporal_fill,
                        ),
                    ),
                    (
                        "fc1",
                        co.Conv3d(dim_in, dim_fc, 1, bias=True),
                    ),
                    (
                        "fc1_act",
                        nn.ReLU()
                        if relu_act
                        else Swish(),  # nn.SELU is the same as Swish
                    ),
                    (
                        "fc2",
                        co.Conv3d(dim_fc, dim_in, 1, bias=True),
                    ),
                    (
                        "fc2_sig",
                        nn.Sigmoid(),
                    ),
                ]
            )
        ),
        reduce="mul",
    )


class CoSeAlt(co.CoModule, nn.Module):
    """Continual Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(
        self,
        window_size: int,
        dim_in: int,
        ratio: float,
        relu_act: bool = True,
        scope="frame",
        temporal_fill="zeros",
    ):
        """
        Args:
            window_size (int): the window size to average over
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
        """
        super(CoSeAlt, self).__init__()
        self.avg_pool = {
            "clip": lambda: co.AdaptiveAvgPool3d(
                output_size=(1, 1, 1),
                kernel_size=window_size,
                temporal_fill=temporal_fill,
            ),
            "frame": lambda: co.forward_stepping(AdaptiveAvgPool2d(output_size=(1, 1))),
        }[scope]()
        dim_fc = _round_width(dim_in, ratio)
        self.fc1 = co.forward_stepping(nn.Conv3d(dim_in, dim_fc, 1, bias=True))
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = co.forward_stepping(nn.Conv3d(dim_fc, dim_in, 1, bias=True))
        self.fc2_sig = nn.Sigmoid()

        self.window_size = window_size
        self.dim_in = dim_in
        self.ratio = ratio
        self.relu_act = relu_act
        self.scope = scope
        self.temporal_fill = temporal_fill

    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x

    def forward_step(self, x: Tensor) -> Tensor:
        x_in = x
        for module in self.children():
            if hasattr(module, "forward_step"):
                x = module.forward_step(x)
            else:
                x = module(x)
            if not isinstance(x, Tensor):
                return None
        return x_in * x

    def forward_steps(self, x: Tensor) -> Tensor:
        x_in = x
        for module in self.children():
            if hasattr(module, "forward_steps"):
                x = module.forward_steps(x)
            else:
                x = module(x)
            if not isinstance(x, Tensor):
                return None
        return x_in * x

    @property
    def delay(self):
        return self.avg_pool.delay

    def clean_state(self):
        self.avg_pool.clean_state()

    def build_from(
        module: SE, window_size: int, scope="frame", temporal_fill="zeros"
    ) -> "CoSeAlt":
        mod = CoSeAlt(
            window_size,
            module.dim_in,
            module.ratio,
            module.relu_act,
            scope,
            temporal_fill,
        )
        mod.load_state_dict(module.state_dict())
        return mod
