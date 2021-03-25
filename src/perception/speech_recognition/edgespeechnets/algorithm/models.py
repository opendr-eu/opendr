# Copyright 2020 Tampere University
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

import torch.nn as nn
import torch.nn.functional as F


class EdgeSpeechNet(nn.Module):

    def __init__(self, target_classes_n):
        super().__init__()
        self.encoder = None
        self._make_encoder()
        self.decoder = nn.Linear(in_features=45, out_features=target_classes_n)

    def _make_encoder(self):
        layers = []
        for entry in self.__class__.architecture:
            layer, kwargs = entry
            layers.append(layer(**kwargs))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.squeeze(2).squeeze(2)
        x = self.decoder(x)
        return F.log_softmax(x, dim=1)


class ESNConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              bias=False,
                              padding=1,
                              padding_mode=padding_mode)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ESNResBlock(nn.Module):

    def __init__(self, in_out_channels, mid_channels, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.prebn = nn.BatchNorm2d(in_out_channels)
        self.conv1 = ESNConv2d(in_channels=in_out_channels, out_channels=mid_channels)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.conv2 = ESNConv2d(in_channels=mid_channels, out_channels=in_out_channels)

    def forward(self, x):
        residual_input = x
        if not self.first:
            x = self.prebn(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x += residual_input
        return x


class EdgeSpeechNetA(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 39}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 20, "first": True}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 15}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 45}),
    ]


class EdgeSpeechNetB(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 30}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8, "first": True}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 10}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 45}),
    ]


class EdgeSpeechNetC(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 24}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6, "first": True}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 12}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 5}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 2}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 45}),
    ]


class EdgeSpeechNetD(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 45}),
        (nn.AvgPool2d, {"kernel_size": 2}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 30, "first": True}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 33}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 35}),
    ]
