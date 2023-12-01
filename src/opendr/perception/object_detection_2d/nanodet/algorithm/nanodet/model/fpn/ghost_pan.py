# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.ghostnet import GhostBottleneck
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import ConvModule, DepthwiseConvModule


class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand=1,
        kernel_size=5,
        kernel_size_shortcut=None,
        num_blocks=1,
        use_res=False,
        activation="LeakyReLU",
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        kernel_size_shortcut = kernel_size if kernel_size_shortcut is None else kernel_size_shortcut
        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        ) if use_res else nn.Identity()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    dw_kernel_size=kernel_size,
                    kernel_size_shortcut=kernel_size_shortcut,
                    activation=activation,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class GhostPAN(nn.Module):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        reduction_depthwise (bool): Whether to depthwise separable convolution in
            reduction module. Default: False
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        kernel_size_shortcut (int): Kernel size of shortcut module. Default: None, if None equal to kernel_size
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        num_extra_level (int): Number of extra conv layers for more feature levels.
            Default: 0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        activation (str): Activation layer name.
            Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=False,
        reduction_depthwise=False,
        kernel_size=5,
        kernel_size_shortcut=None,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        norm_cfg=dict(type="BN"),
        activation="LeakyReLU",
    ):
        super(GhostPAN, self).__init__()
        assert num_extra_level >= 0
        assert num_blocks >= 1
        kernel_size_shortcut = kernel_size if kernel_size_shortcut is None else kernel_size_shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseConvModule if use_depthwise else ConvModule
        reduction_conv = DepthwiseConvModule if reduction_depthwise else ConvModule

        # build top-down blocks
        modes = ["linear", "bilinear", "bicubic", "trilinear"]
        try:
            self.upsample = nn.Upsample(**upsample_cfg, align_corners=False if upsample_cfg.mode in modes else None)
        except:
            self.upsample = nn.Upsample(**upsample_cfg, align_corners=False if upsample_cfg["mode"] in modes else None)

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(
                reduction_conv(
                    in_channels[idx],
                    out_channels,
                    1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    kernel_size_shortcut=kernel_size_shortcut,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    kernel_size_shortcut=kernel_size_shortcut,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        # extra layers
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        self.num_extra_level = num_extra_level
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.extra_lvl_out_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

    def forward(self, inputs: List[Tensor]):
        """
        Args:
            inputs (List[Tensor]): input features.
        Returns:
            List[Tensor]: multi level features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = [
            reduce(inputs[indx]) for indx, (reduce) in enumerate(self.reduce_layers)
        ]
        # top-down path
        inner_outs = [inputs[-1]]
        for idx, top_down_block in enumerate(self.top_down_blocks):
            reversed_idx = len(self.in_channels) - 1 - idx
            if reversed_idx != 0:
                feat_heigh = inner_outs[0]
                feat_low = inputs[reversed_idx - 1]

                upsample_feat = self.upsample(feat_heigh)

                inner_out = top_down_block(
                    torch.cat([upsample_feat, feat_low], 1)
                )
                inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx, (downsample, bottom_up_block) in enumerate(zip(self.downsamples, self.bottom_up_blocks)):
            if idx != len(self.in_channels) - 1:
                feat_low = outs[-1]
                feat_height = inner_outs[idx + 1]
                downsample_feat = downsample(feat_low)
                out = bottom_up_block(
                    torch.cat([downsample_feat, feat_height], 1)
                )
                outs.append(out)

        # extra layers
        for indx, (extra_in_layer, extra_out_layer) in enumerate(zip(self.extra_lvl_in_conv, self.extra_lvl_out_conv)):
            outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))

        return outs
