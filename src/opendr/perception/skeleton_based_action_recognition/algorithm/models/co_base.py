"""
Modified based on: https://github.com/LukasHedegaard/continual-skeletons
"""

import math
from collections import OrderedDict
from typing import Sequence

import continual as co
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from numpy import prod
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from logging import getLogger
import pytorch_lightning as pl

from opendr.perception.skeleton_based_action_recognition.algorithm.graphs.nturgbd import NTUGraph
from opendr.perception.skeleton_based_action_recognition.algorithm.graphs.kinetics import (
    KineticsGraph,
)

logger = getLogger(__name__)


class CoModelBase(pl.LightningModule, co.Sequential):
    def __init__(
        self,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph_type="ntu",
        sequence_len: int = 300,
        num_classes=60,
        loss_name="cross_entropy",
        forward_mode: str = "clip",  # choices=["clip", "frame"]
        predict_after_frames: int = 0,
        continual_temporal_fill: str = "zeros",  # choices=["zeros", "replicate"]
        pool_size: int = -1,
        pool_padding: int = -1,
    ):
        pl.LightningModule.__init__(self)
        self.forward_mode = forward_mode
        self.predict_after_frames = predict_after_frames
        self.continual_temporal_fill = continual_temporal_fill
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.graph_type = graph_type
        self.sequence_len = sequence_len
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.input_shape = (in_channels, sequence_len, num_point, num_person)
        if graph_type == "ntu" or num_point == 25:
            self.graph = NTUGraph()
        elif graph_type == "openpose" or num_point == 18:
            self.graph = KineticsGraph()

        co.Sequential.__init__(self)

    def on_init_end(self):
        # Shapes from Dataset:
        # num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, V, S) = self.input_shape

        def reshape1_fn(x):
            return x.permute(0, 3, 2, 1).contiguous().view(-1, S * V * C_in)

        reshape1 = co.Lambda(reshape1_fn)
        data_bn = nn.BatchNorm1d(S * C_in * V)

        def reshape2_fn(x):
            return x.view(-1, S, V, C_in).permute(0, 1, 3, 2).contiguous().view(-1, C_in, V)

        reshape2 = co.Lambda(reshape2_fn)

        spatial_pool = co.Lambda(lambda x: x.view(-1, S, 256, V).mean(3).mean(1))

        pool_size = self.pool_size
        if pool_size == -1:
            pool_size = math.ceil(
                (T - self.receptive_field + 2 * self.padding[0] + 1) / self.stride[0]
            )

        pool_padding = self.pool_padding
        if pool_padding == -1:
            pool_padding = pool_size - math.ceil(
                (T - self.receptive_field + self.padding[0] + 1) / self.stride[0]
            )
        pool = co.AvgPool1d(pool_size, stride=1, padding=max(0, pool_padding))

        fc = co.Linear(256, self.num_classes, channel_dim=1)

        squeeze = co.Lambda(lambda x: x.squeeze(-1), takes_time=True, forward_step_only_fn=lambda x: x)

        # Initialize weights
        init_weights(data_bn, bs=1)
        init_weights(fc, bs=self.num_classes)

        # Add blocks sequentially
        co.Sequential.__init__(
            self,
            OrderedDict(
                [
                    ("reshape1", reshape1),
                    ("data_bn", data_bn),
                    ("reshape2", reshape2),
                    ("layers", self.layers),
                    ("spatial_pool", spatial_pool),
                    ("pool", pool),
                    ("fc", fc),
                    ("squeeze", squeeze),
                ]
            ),
        )

        if self.forward_mode == "frame":
            self.call_mode = "forward_steps"  # Set continual forward mode

        logger.info(f"Input shape (C, T, V, S) = {self.input_shape}")
        logger.info(f"Receptive field {self.receptive_field}")
        logger.info(f"Init frames {self.receptive_field - 2 * self.padding[0] - 1}")
        logger.info(f"Pool size {pool_size}")
        logger.info(f"Stride {self.stride[0]}")
        logger.info(f"Padding {self.padding[0]}")
        logger.info(f"Using Continual {self.call_mode}")

        if self.forward_mode == "frame":
            (num_channels, num_frames, num_vertices, num_skeletons) = self.input_shape

            # A new output is created every `self.stride` frames.
            self.input_shape = (num_channels, self.stride, num_vertices, num_skeletons)

    def warm_up(self, dummy, sample: Sequence[int], *args, **kwargs):
        # Called prior to profiling

        if self.forward_mode == "clip":
            return

        self.clean_state()

        N, C, T, S, V = sample.shape

        self._current_input_shape = (N, C, S, V)

        init_frames = self.receptive_field - self.padding - 1
        init_data = torch.randn((N, C, init_frames, S, V)).to(device=self.device)
        for i in range(init_frames):
            self.forward_step(init_data[:, :, i])

    def clean_state_on_shape_change(self, shape):
        if getattr(self, "_current_input_shape", None) != shape:
            self._current_input_shape = shape
            self.clean_state()

    def forward(self, input):
        if self.forward_mode == "clip":
            ret = super().forward(input)
        else:
            assert self.forward_mode == "frame"
            N, C, T, S, V = input.shape
            self.clean_state_on_shape_change((N, C, S, V))

            if not self.profile_model:
                self.clean_state()

            ret = super().forward_steps(input, update_state=True, pad_end=False)

        if len(getattr(ret, "shape", (0,))) == 3:
            ret = ret[:, :, 0]  # the rest may be end-padding
        return ret

    def forward_step(self, input, update_state=True):
        self.clean_state_on_shape_change(input.shape)
        return super().forward_step(input, update_state)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        N, C, T, S, V = input.shape
        self.clean_state_on_shape_change((N, C, S, V))
        return super().forward_steps(input, pad_end, update_state)

    def map_state_dict(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
    ) -> "OrderedDict[str, Tensor]":
        def map_key(k: str):
            # Handle "layers.layer2.0.1.gcn.g_conv.0.weight" -> "layers.layer2.gcn.g_conv.0.weight"
            k = k.replace("0.1.", "")

            # Handle "layers.layer8.0.0.residual.t_conv.weight" ->layers.layer8.residual.t_conv.weight'
            k = k.replace("0.0.residual", "residual")
            return k

        long_keys = nn.Module.state_dict(self, keep_vars=True).keys()

        if len(long_keys - state_dict.keys()):
            short2long = {map_key(k): k for k in long_keys}
            state_dict = OrderedDict(
                [(short2long[k], v) for k, v in state_dict.items() if strict or k in short2long]
            )
        return state_dict

    def map_loaded_weights(self, file, loaded_state_dict):
        return self.map_state_dict(loaded_state_dict)

    def training_step(self, batch, batch_idx):
        x, y = _unpack_batch(batch)
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log("train/loss", loss)
        self.log("train/acc", _accuracy(x, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = _unpack_batch(batch)
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log("val/loss", loss)
        self.log("val/acc", _accuracy(x, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = _unpack_batch(batch)
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log("test/loss", loss)
        self.log("test/acc", _accuracy(x, y))
        return loss


def _unpack_batch(batch):
    if len(batch) == 3:
        x, y, _ = batch
    return (x, y)


def _accuracy(x: Tensor, y: Tensor):
    return torch.sum(x.argmax(dim=1) == y) / len(y)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, bn_momentum=0.1, *args, **kwargs):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            init_weights(self.g_conv[i], bs=self.num_subset)

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A * self.graph_attn
        sum_ = None
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            sum_ = z + sum_ if sum_ is not None else z
        sum_ = self.bn(sum_)
        sum_ += self.gcn_residual(x)
        return self.relu(sum_)


def CoGraphConvolution(in_channels, out_channels, A, bn_momentum=0.1):
    return co.forward_stepping(GraphConvolution(in_channels, out_channels, A, bn_momentum))


class TemporalConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        padding=4,
    ):
        super(TemporalConvolution, self).__init__()

        self.padding = padding
        self.t_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.padding, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        init_weights(self.t_conv, bs=1)
        init_weights(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


def CoTemporalConvolution(
    in_channels,
    out_channels,
    kernel_size=9,
    padding=0,
    stride=1,
) -> co.Sequential:

    if padding == "equal":
        padding = int((kernel_size - 1) / 2)

    t_conv = co.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(kernel_size, 1),
        padding=(padding, 0),
        stride=(stride, 1),
    )

    bn = nn.BatchNorm2d(out_channels)

    init_weights(t_conv, bs=1)
    init_weights(bn, bs=1)

    seq = []
    seq.append(("t_conv", t_conv))
    seq.append(("bn", bn))
    return co.Sequential(OrderedDict(seq))


class SpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        temporal_kernel_size=9,
        temporal_padding=-1,
        GraphConv=GraphConvolution,
        TempConv=TemporalConvolution,
    ):
        super(SpatioTemporalBlock, self).__init__()
        equal_padding = int((temporal_kernel_size - 1) / 2)
        if temporal_padding < 0:
            temporal_padding = equal_padding
            self.residual_shrink = None
        else:
            assert temporal_padding <= equal_padding
            self.residual_shrink = equal_padding - temporal_padding
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TempConv(
            out_channels,
            out_channels,
            stride=stride,
            kernel_size=temporal_kernel_size,
            padding=temporal_padding,
        )
        self.relu = nn.ReLU()
        if not residual:
            self.residual = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = unity
        else:
            self.residual = TempConv(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x):
        z = self.tcn(self.gcn(x))

        if self.residual_shrink:
            # Centered residuals:
            # If temporal zero-padding is removed, the feature-map shrinks at every temporal conv
            # The residual should shrink correspondingly, i.e. (kernel_size - 1) / 2) on each side
            r = self.residual(x[:, :, self.residual_shrink:-self.residual_shrink])
        else:
            r = self.residual(x)

        return self.relu(z + r)


def CoSpatioTemporalBlock(
    in_channels,
    out_channels,
    A,
    stride=1,
    residual=True,
    window_size=1,
    padding=0,
    CoGraphConv=CoGraphConvolution,
    CoTempConv=CoTemporalConvolution,
):
    window_size = int(window_size)  # Currently unused. Could be used BN momentum

    gcn = CoGraphConv(in_channels, out_channels, A, bn_momentum=0.1)
    tcn = CoTempConv(
        out_channels,
        out_channels,
        stride=stride,
        padding=padding,
    )
    relu = torch.nn.ReLU()

    if not residual:
        return co.Sequential(OrderedDict([("gcn", gcn), ("tcn", tcn), ("relu", relu)]))

    if (in_channels == out_channels) and (stride == 1):
        return co.Sequential(
            co.Residual(
                co.Sequential(OrderedDict([("gcn", gcn), ("tcn", tcn)])),
                residual_shrink=True,
            ),
            relu,
        )

    residual = CoTempConv(in_channels, out_channels, kernel_size=1, stride=stride)

    residual_shrink = tcn.receptive_field - 2 * tcn.padding[0] != 1

    delay = tcn.delay // stride
    if residual_shrink:
        delay = delay // 2

    return co.Sequential(
        co.BroadcastReduce(
            co.Sequential(
                OrderedDict(
                    [
                        ("residual", residual),
                        ("align", co.Delay(delay, auto_shrink=residual_shrink)),
                    ]
                )
            ),
            co.Sequential(OrderedDict([("gcn", gcn), ("tcn", tcn)])),
            auto_delay=False,
        ),
        relu,
    )


def init_weights(module_, bs=1):
    if isinstance(module_, _ConvNd):
        nn.init.constant_(module_.bias, 0)
        if bs == 1:
            nn.init.kaiming_normal_(module_.weight, mode="fan_out")
        else:
            nn.init.normal_(
                module_.weight,
                0,
                math.sqrt(2.0 / (prod(module_.weight.size()) * bs)),
            )
    elif isinstance(module_, _BatchNorm):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2.0 / bs))


def zero(x):
    return 0


def unity(x):
    return x
