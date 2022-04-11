# Copyright 2020-2022 OpenDR European Project
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
"""
This code is modified from https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class LocalFeatureExtractor(nn.Module):

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index

        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()

        self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)

        self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)

        self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)

        self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)

    def forward(self, x):

        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]

        out_1 = self.conv1_1(patch_11)
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)

        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)

        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)

        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
