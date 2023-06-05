# Copyright 2020-2023 OpenDR European Project
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


class VGG_1080p_64(nn.Module):
    def __init__(self):
        super(VGG_1080p_64, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=0)
        self.conv_dec = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=13, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.conv_dec(x)
        return x


class VGG_720p_64(nn.Module):
    def __init__(self):
        super(VGG_720p_64, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, stride=4, padding=1)
        self.conv_dec = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=8, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv_dec(x)
        return x
