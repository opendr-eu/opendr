# Copyright 2020-2021 OpenDR European Project
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_filter, kernel_size, padding, dropout, pooling, expand_right):
        super(ResidualBlock, self).__init__()
        # code is adapted from https://github.com/fernandoandreotti/cinc-challenge2017

        left_branch = [
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding)
        ]

        right_branch = []
        if pooling:
            left_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if expand_right:
                right_branch.append(nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=1, stride=1))
                right_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                right_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))

        else:
            if expand_right:
                right_branch.append(nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=1))

        if len(right_branch) == 0:
            self.right_branch = None
        elif len(right_branch) == 1:
            self.right_branch = right_branch[0]
        else:
            self.right_branch = nn.Sequential(*right_branch)

        self.left_branch = nn.Sequential(*left_branch)
        self.initialize()
        self.expand_right = expand_right
        self.pooling = pooling

    def forward(self, x):
        if self.right_branch is not None:
            left = self.left_branch(x)
            right = self.right_branch(x)
            x = left + right
        else:
            x = self.left_branch(x) + x

        return x

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)

    def get_parameters(self):
        bn_params = list(self.left_branch[0].parameters()) +\
            list(self.left_branch[4].parameters())
        other_params = list(self.left_branch[3].parameters()) +\
            list(self.left_branch[7].parameters())

        if self.expand_right:
            if self.pooling:
                other_params.extend(list(self.right_branch[0].parameters()))
            else:
                other_params.extend(list(self.right_branch.parameters()))

        return bn_params, other_params


class ResNetPreprocessing(nn.Module):
    def __init__(self, in_channels, n_filter=64, kernel_size=15, padding=7, dropout=0.5):
        super(ResNetPreprocessing, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU()
        )

        self.block2_1 = nn.Sequential(
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block2_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        residual_blocks = []
        pooling = False
        filter_multiplier = 1
        in_channels = n_filter

        for layer_idx in range(15):
            if layer_idx % 4 == 0 and layer_idx > 0:
                filter_multiplier += 1
                expand_right = True
            else:
                expand_right = False

            residual_blocks.append(ResidualBlock(in_channels,
                                                 n_filter * filter_multiplier,
                                                 kernel_size,
                                                 padding,
                                                 dropout,
                                                 pooling,
                                                 expand_right))
            pooling = not pooling
            in_channels = n_filter * filter_multiplier

        self.residual_blocks = nn.Sequential(*residual_blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_1(x) + self.block2_2(x)
        x = self.residual_blocks(x)
        return x

    def get_parameters(self):
        bn_params, other_params = [], []

        # block 1
        bn_params.extend(list(self.block1[1].parameters()))
        other_params.extend(list(self.block1[0].parameters()))

        # block 2_1
        bn_params.extend(list(self.block2_1[1].parameters()))
        other_params.extend(list(self.block2_1[0].parameters()))
        other_params.extend(list(self.block2_1[4].parameters()))

        # residual blocks
        for layer in self.residual_blocks:
            bn, other = layer.get_parameters()
            bn_params.extend(bn)
            other_params.extend(other)

        return bn_params, other_params


class GRU(nn.Module):
    def __init__(self, in_channels, series_length, recurrent_unit, n_class, dropout):
        super(GRU, self).__init__()
        # resnet preprocessing block
        self.resnet_block = ResNetPreprocessing(in_channels=in_channels)

        # gru block
        in_channels, series_length = self.compute_intermediate_dimensions(in_channels, series_length)
        self.gru_layer = nn.GRU(input_size=in_channels, hidden_size=recurrent_unit, batch_first=True)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(in_features=recurrent_unit, out_features=512),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(in_features=512, out_features=n_class))

    def forward(self, x):
        x = self.resnet_block(x)
        x = x.transpose(-1, -2)
        output = self.gru_layer(x)[0]
        x = output[:, -1, :]
        x = self.classifier(x)
        return x

    def compute_intermediate_dimensions(self, in_channels, series_length):
        with torch.no_grad():
            x = torch.randn(1, in_channels, series_length)
            y = self.resnet_block(x)
            n_channels = y.size(1)
            length = y.size(2)
            return n_channels, length

    def get_parameters(self):
        bn_params, other_params = self.resnet_block.get_parameters()
        other_params.extend(list(self.gru_layer.parameters()))
        other_params.extend(list(self.classifier.parameters()))
        return bn_params, other_params
