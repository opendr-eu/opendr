"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolkit with permission from the author.
"""


from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.conv import _ConvNd, _reverse_repeat_tuple, _size_3_t, _triple

from .utils import warn_once_if
from logging import getLogger

from .utils import FillMode

State = Tuple[Tensor, int]

logger = getLogger(__name__)


class ConvCo3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: FillMode = "zeros",
        temporal_fill: FillMode = "replicate",
    ):
        r"""Applies a recursive 3D convolution over an input signal composed of several input
        planes.

        Assuming an input of shape `(B, C, T, H, W)`, it computes the convolution over one temporal instant `t` at a time
        where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

        `forward3d` operates identically to `Conv3d.forward`

        `forward`   takes an input of shape `(B, C, H, W)`, and computes a single-frame output (B, C', H', W') based on
                    its internal state. On the first execution, the state is initialised with either ``'zeros'``
                    (corresponding to a zero padding of kernel_size[0]-1) or with a `'replicate'`` of the first frame
                    depending on the choice of `temporal_fill`. `forward` also supports a functional-style exercution,
                    by passing a `prev_state` explicitely as parameters, and by optionally returning the updated
                    `next_state` via the `return_next_state` parameter.
                    NB: The output when recurrently applying forward will be delayed by the `kernel_size[0] - 1`.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution.
                NB: stride > 1 over the first channel is notsupported. Default: 1
            padding (int or tuple, optional): Zero-padding added to all three sides of the input.
                NB: padding over the first channel is not supported. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements.
                NB: dilation > 1 over the first channel is not supported. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            temporal_fill (string, optional): ``'zeros'`` or ``'replicate'`` (= "boring video").
                `temporal_fill` determines how state is initialised and which padding is applied during `forward3d`
                along the temporal dimension. Default: ``'replicate'``

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                            The values of these weights are sampled from
                            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
            bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                            then the values of these weights are
                            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
            state (List[Tensor]):  a running buffer of partial computations from previous frames which are used for
                            the calculation of subsequent outputs.

        """
        kernel_size = _triple(kernel_size)

        padding = _triple(padding)
        warn_once_if(
            cond=padding[0] != 0,
            msg=(
                "Padding along the temporal dimension only affects the computation in `forward3d`. "
                "In `forward` it is omitted."
            )
        )

        stride = _triple(stride)
        assert stride[0] == 1, "Temporal stride > 1 is not supported currently."

        dilation = _triple(dilation)
        assert dilation[0] == 1, "Temporal dilation > 1 is not supported currently."

        super(ConvCo3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
            (self.kernel_size[0] - 1, *self.padding[1:]),
            2
            # (0, *self.padding[1:]), 2
        )

        assert temporal_fill in {"zeros", "replicate"}
        self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]
        # init_state is called in `_forward`

    def init_state(self, first_output: Tensor,) -> State:
        padding = self.make_padding(first_output)
        state_buffer = padding.repeat(self.kernel_size[0] - 1, 1, 1, 1, 1, 1)
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

    @staticmethod
    def from_3d(
        module: torch.nn.Conv3d, temporal_fill: FillMode = "replicate"
    ) -> "ConvCo3d":
        stride = (1, *module.stride[1:])
        dilation = (1, *module.dilation[1:])
        for shape, name in zip([stride, dilation], ["stride", "dilation"]):
            prev_shape = getattr(module, name)
            if shape != prev_shape:
                logger.warning(
                    f"Using {name} = {shape} for RConv3D (converted from {prev_shape})"
                )

        rmodule = ConvCo3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=stride,
            padding=module.padding,
            dilation=dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            temporal_fill=temporal_fill,
        )
        with torch.no_grad():
            rmodule.weight.copy_(module.weight)
            if module.bias is not None:
                rmodule.bias.copy_(module.bias)
        return rmodule

    def forward(self, input: Tensor, update_state=True) -> Tensor:
        output, (new_buffer, new_index) = self._forward(input, self.get_state())
        if update_state:
            self.state_buffer = new_buffer
            self.state_index = new_index
        return output

    def _forward(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        assert len(input.shape) == 4, "Only a single frame should be passed at a time."

        # B, C, H, W -> B, C, 1, H, W
        x = input.unsqueeze(2)

        if self.padding_mode == "zeros":
            x = F.conv3d(
                input=x,
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=(self.kernel_size[0] - 1, *self.padding[1:]),
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            x = F.conv3d(
                input=F.pad(
                    x, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=(self.kernel_size[0] - 1, 0, 0),
                dilation=self.dilation,
                groups=self.groups,
            )

        x_out, x_rest = x[:, :, 0].clone(), x[:, :, 1:]

        # Prepare previous state
        buffer, index = prev_state or self.init_state(x_rest)

        tot = len(buffer)
        for i in range(tot):
            x_out += buffer[(i + index) % tot, :, :, tot - i - 1]

        if self.bias is not None:
            x_out += self.bias[None, :, None, None]

        # Update next state
        next_buffer = buffer.clone() if self.training else buffer.detach()
        next_buffer[index] = x_rest
        next_index = (index + 1) % tot
        return x_out, (next_buffer, next_index)

    def forward3d(self, input: Tensor):
        assert (
            len(input.shape) == 5
        ), "A tensor of size B,C,T,H,W should be passed as input."
        T = input.shape[2]

        pad_start = [self.make_padding(input[:, :, 0]) for _ in range(self.padding[0])]
        inputs = [input[:, :, t] for t in range(T)]
        pad_end = [self.make_padding(input[:, :, -1]) for _ in range(self.padding[0])]

        # Recurrently pass through, updating state
        outs = []
        for t, i in enumerate([*pad_start, *inputs]):
            o, (self.state_buffer, self.state_index) = self._forward(
                i, self.get_state()
            )
            if self.kernel_size[0] - 1 <= t:
                outs.append(o)

        # Don't save state for the end-padding
        tmp_buffer, tmp_index = self.get_state()
        for t, i in enumerate(pad_end):
            o, (tmp_buffer, tmp_index) = self._forward(i, (tmp_buffer, tmp_index))
            outs.append(o)

        if len(outs) > 0:
            outs = torch.stack(outs, dim=2)
        else:
            outs = torch.tensor([])
        return outs

    @property
    def delay(self):
        return self.kernel_size[0] - 1
