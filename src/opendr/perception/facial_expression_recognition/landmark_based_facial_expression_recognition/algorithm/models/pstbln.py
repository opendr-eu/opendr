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

import math
import numpy as np
import torch
import torch.nn as nn


def weights_init(module_, bs=1):
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode='fan_out')
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(module_.weight, 0,
                        math.sqrt(2. / (module_.weight.size(0) * module_.weight.size(1) * module_.weight.size(2) * bs)))
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2. / bs))


class BilinearMapping(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, cuda_):
        super(BilinearMapping, self).__init__()
        self.cuda_ = cuda_
        self.num_subset = 3
        self.rand_graph = nn.Parameter(torch.from_numpy(np.random.rand(self.num_subset, num_point,
                                                                       num_point).astype(np.float32)))
        nn.init.constant_(self.rand_graph, 1e-6)
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            weights_init(self.g_conv[i], bs=self.num_subset)

        if in_channels != out_channels:
            self.bln_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            weights_init(self.bln_residual[0], bs=1)
            weights_init(self.bln_residual[1], bs=1)
        else:
            self.bln_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.rand_graph
        if self.cuda_:
            A = A.cuda(x.get_device())
        hidden_ = None
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.bln_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                padding=(pad, 0), stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        weights_init(self.t_conv, bs=1)
        weights_init(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


class ST_BLN_block(nn.Module):
    def __init__(self, topology, blocksize, layer_ind, num_point, cuda_=False, stride=1, residual=True):
        super(ST_BLN_block, self).__init__()

        if layer_ind == 0:
            in_channels = 2
            residual = False
        else:
            in_channels = topology[layer_ind - 1] * blocksize
        out_channels = topology[layer_ind] * (blocksize)
        if layer_ind == 4 or layer_ind == 7:
            stride = 2

        self.bln = BilinearMapping(in_channels, out_channels, num_point, cuda_)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn(self.bln(x)) + self.residual(x)
        return self.relu(x)


class PSTBLN(nn.Module):
    def __init__(self, topology, blocksize, num_class, num_point, num_person=1, in_channels=2,
                 cuda_=False):
        super(PSTBLN, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        weights_init(self.data_bn, bs=1)
        self.dropout_ = nn.Dropout(p=0.2)
        self.topology = topology
        self.cuda_ = cuda_
        self.num_layers = len(self.topology)
        self.block_size = blocksize
        self.layers = nn.ModuleDict(
            {'l{}'.format(i): ST_BLN_block(self.topology, self.block_size, i, num_point=num_point, cuda_=self.cuda_)
             for i in range(self.num_layers)})

        self.fc = nn.Linear(self.block_size * topology[-1], num_class)  # the normal one
        weights_init(self.fc, bs=num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_layers):
            x = self.dropout_(x)
            x = self.layers['l' + str(i)](x)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
