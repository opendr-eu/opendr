"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolit with permission from the author.
"""

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn.modules.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool2d,
    AvgPool3d,
    MaxPool2d,
    MaxPool3d,
    _triple,
)

from logging import getLogger

from .utils import FillMode

State = Tuple[Tensor, int]
Pool2D = Union[AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d]


logger = getLogger(__name__)

__all__ = [
    "AvgPoolCo3d",
    "MaxPoolCo3d",
    "AdaptiveAvgPoolCo3d",
    "AdaptiveMaxPoolCo3d",
    "convert_avgpool3d",
    "convert_maxpool3d",
    "convert_adaptiveavgpool3d",
    "convert_adaptivemaxpool3d",
]


def RecursivelyWindowPooled(cls: Pool2D) -> torch.nn.Module:  # noqa: C901
    """Wraps a pooling module to create a recursive version which pools across execusions

    Args:
        cls (Pool2D): A 2D pooling Module
    """
    assert cls in {AdaptiveAvgPool2d, MaxPool2d, AvgPool2d, AdaptiveMaxPool2d}

    class RePooled(cls):
        def __init__(
            self,
            window_size: int,
            temporal_fill: FillMode = "replicate",
            temporal_dilation: int = 1,
            *args,
            **kwargs,
        ):
            assert window_size > 0
            assert temporal_fill in {"zeros", "replicate"}
            self.window_size = window_size
            self.temporal_dilation = temporal_dilation
            self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
                temporal_fill
            ]
            super(RePooled, self).__init__(*args, **kwargs)

            self.temporal_pool = (
                AdaptiveAvgPool1d
                if "avg" in str(cls.__name__).lower()
                else AdaptiveMaxPool1d
            )(1)

            if self.temporal_dilation > 1:
                self.frame_index_selection = torch.tensor(
                    range(0, self.window_size, self.temporal_dilation)
                )

            # state is initialised in self.forward

        def init_state(self, first_output: Tensor,) -> State:
            padding = self.make_padding(first_output)
            state_buffer = torch.stack(
                [padding for _ in range(self.window_size)], dim=0
            )
            state_index = 0
            if not hasattr(self, "state_buffer"):
                self.register_buffer("state_buffer", state_buffer, persistent=False)
            return state_buffer, state_index

        def clean_state(self):
            self.state_buffer = None
            self.state_index = None

        def get_state(self):
            if (
                hasattr(self, "state_buffer") and
                self.state_buffer is not None and
                hasattr(self, "state_index") and
                self.state_buffer is not None
            ):
                return (self.state_buffer, self.state_index)
            else:
                return None

        def forward(self, input: Tensor) -> Tensor:
            output, (self.state_buffer, self.state_index) = self._forward(
                input, self.get_state()
            )
            return output

        def _forward(self, input: Tensor, prev_state: State,) -> Tuple[Tensor, State]:
            assert (
                len(input.shape) == 4
            ), "Only a single frame should be passed at a time."

            pooled_frame = super(RePooled, self).forward(input)

            if prev_state is None:
                buffer, index = self.init_state(pooled_frame)
            else:
                buffer, index = prev_state

            buffer[index] = pooled_frame

            if self.temporal_dilation == 1:
                frame_selection = buffer
            else:
                frame_selection = buffer.index_select(
                    dim=0, index=self.frame_index_selection
                )

            # Pool along temporal dimension
            T, B, C, H, W = frame_selection.shape
            x = frame_selection.permute(1, 3, 4, 2, 0)  # B, H, W, C, T
            x = x.reshape(B * H * W, C, T)
            x = self.temporal_pool(x)
            x = x.reshape(B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            pooled_window = x

            new_index = (index + 1) % self.window_size
            new_buffer = buffer.clone() if self.training else buffer.detach()

            return pooled_window, (new_buffer, new_index)

        def forward3d(self, input: Tensor):
            """ If input.shape[2] == self.window_size, a global pooling along temporal dimension is performed
                Otherwise, the pooling is performed per frame
            """
            assert (
                len(input.shape) == 5
            ), "A tensor of size B,C,T,H,W should be passed as input."

            outs = []
            for t in range(input.shape[2]):
                o = self.forward(input[:, :, t])
                if self.window_size - 1 <= t:
                    outs.append(o)

            if len(outs) == 0:
                return torch.tensor([])

            if input.shape[2] == self.window_size:
                # In order to be compatible with downstream forward3d, select only last frame
                # This corrsponds to the regular global pool
                return outs[-1].unsqueeze(2)

            else:
                return torch.stack(outs, dim=2)

    RePooled.__doc__ = f"""
    Recursive {cls.__name__}

    Pooling results are stored between `forward` exercutions and used to pool subsequent
    inputs along the temporal dimension with a spacified `window_size`.
    Example: For `window_size = 3`, the two previous results are stored and used for pooling.
    `temporal_fill` determines whether to initialize the state with a ``'replicate'`` of the
    output of the first execution or with with ``'zeros'``.

    Parent doc:
    {cls.__doc__}
    """

    return RePooled


AvgPoolCo3d = RecursivelyWindowPooled(AvgPool2d)
MaxPoolCo3d = RecursivelyWindowPooled(MaxPool2d)
AdaptiveAvgPoolCo3d = RecursivelyWindowPooled(AdaptiveAvgPool2d)
AdaptiveMaxPoolCo3d = RecursivelyWindowPooled(AdaptiveMaxPool2d)


def convert_avgpool3d(
    instance: AvgPool3d,
    window_size: int = None,  # Not used: only there to satisfy interface
    temporal_fill: FillMode = "replicate",
):
    kernel_size = _triple(instance.kernel_size)
    padding = _triple(instance.padding)
    stride = _triple(instance.stride)
    assert padding[0] == 0, "Cannot convert AvgPool3d with padding[0] != 0"
    assert stride[0] == 1, "Cannot convert AvgPool3d with stride[0] != 1"
    return AvgPoolCo3d(
        window_size=kernel_size[0],
        temporal_fill=temporal_fill,
        kernel_size=kernel_size[1:],
        stride=stride[1:],
        padding=padding[1:],
        ceil_mode=instance.ceil_mode,
        count_include_pad=instance.count_include_pad,
        divisor_override=instance.divisor_override,
    )


def convert_maxpool3d(
    instance: MaxPool3d,
    window_size: int = None,  # Not used: only there to satisfy interface
    temporal_fill: FillMode = "replicate",
):
    kernel_size = _triple(instance.kernel_size)
    padding = _triple(instance.padding)
    stride = _triple(instance.stride)
    dilation = _triple(instance.dilation)
    assert padding[0] == 0, "Cannot convert MaxPool3d with padding[0] != 0"
    assert stride[0] == 1, "Cannot convert MaxPool3d with stride[0] != 1"
    assert dilation[0] == 1, "Cannot convert MaxPool3d with dilation[0] != 1"
    assert (
        instance.return_indices is False
    ), "return_indices currently not supported for MaxPool3d"
    return MaxPoolCo3d(
        window_size=kernel_size[0],
        temporal_fill=temporal_fill,
        kernel_size=kernel_size[1:],
        stride=stride[1:],
        padding=padding[1:],
        dilation=dilation[1:],
        return_indices=instance.return_indices,
        ceil_mode=instance.ceil_mode,
    )


def convert_adaptiveavgpool3d(
    instance: AdaptiveAvgPool3d,
    window_size: int,
    temporal_fill: FillMode = "replicate",
):
    assert (
        instance.output_size[0] == 1
    ), "Cannot convert AdaptiveAvgPool3d without output_size[0] != 1"
    return AdaptiveAvgPoolCo3d(
        window_size=window_size,
        temporal_fill=temporal_fill,
        output_size=instance.output_size[1:],
    )


def convert_adaptivemaxpool3d(
    instance: AdaptiveMaxPool3d,
    window_size: int,
    temporal_fill: FillMode = "replicate",
):
    assert (
        instance.output_size[0] == 1
    ), "Cannot convert AdaptiveMaxPool3d without output_size[0] != 1"
    assert (
        instance.return_indices is False
    ), "return_indices currently not supported for AdaptiveMaxPool3d"
    return AdaptiveAvgPoolCo3d(
        window_size=window_size,
        temporal_fill=temporal_fill,
        output_size=instance.output_size,
    )
