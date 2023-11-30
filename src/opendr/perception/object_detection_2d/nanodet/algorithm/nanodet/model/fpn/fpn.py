# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import ConvModule
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.init_weights import xavier_init


class FPN(nn.Module):
    """Feature proposal network

        Args:
            in_channels (List[int]): Number of input channels per scale.
            out_channels (int): Number of output channels (used at each scale)
            num_outs (int): Number of output scales.
            start_level (int): Index of the start input backbone level used to
                build the feature pyramid. Default: 0.
            end_level (int): Index of the end input backbone level (exclusive) to
                build the feature pyramid. Default: -1, which means the last level.
            norm_cfg (dict): Config dict for normalization layer. Default: None.
            activation (str): Config dict for activation layer in ConvModule.
                Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs: List[Tensor]):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2.0, mode="bilinear"
            )

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return outs
