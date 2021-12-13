"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from opendr.perception.skeleton_based_action_recognition.algorithm.graphs.nturgbd import NTUGraph
from opendr.perception.skeleton_based_action_recognition.algorithm.graphs.kinetics import KineticsGraph


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


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_):
        super(GraphConvolution, self).__init__()
        self.cuda_ = cuda_
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            weights_init(self.g_conv[i], bs=self.num_subset)

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A * self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.t_conv, bs=1)
        weights_init(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


class ST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_=False, stride=1, residual=True):
        super(ST_GCN_block, self).__init__()

        self.gcn = GraphConvolution(in_channels, out_channels, A, cuda_)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, graph_type, cuda_=False):
        super(STGCN, self).__init__()

        if graph_type == 'ntu' or num_point == 25:
            self.graph = NTUGraph()
        elif graph_type == 'openpose' or num_point == 18:
            self.graph = KineticsGraph()

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        weights_init(self.data_bn, bs=1)

        self.layers = nn.ModuleDict(
            {'layer1': ST_GCN_block(in_channels, 64, A, cuda_, residual=False),
             'layer2': ST_GCN_block(64, 64, A, cuda_),
             'layer3': ST_GCN_block(64, 64, A, cuda_),
             'layer4': ST_GCN_block(64, 64, A, cuda_),
             'layer5': ST_GCN_block(64, 128, A, cuda_, stride=2),
             'layer6': ST_GCN_block(128, 128, A, cuda_),
             'layer7': ST_GCN_block(128, 128, A, cuda_),
             'layer8': ST_GCN_block(128, 256, A, cuda_, stride=2),
             'layer9': ST_GCN_block(256, 256, A, cuda_),
             'layer10': ST_GCN_block(256, 256, A, cuda_)}
        )

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(len(self.layers)):
            x = self.layers['layer' + str(i+1)](x)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)
