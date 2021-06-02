# MIT License
#
# Copyright (c) 2017 liukuang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model_utils


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

    def get_parameters(self,):
        bn_params = list(self.bn1.parameters()) + list(self.bn2.parameters())
        other_params = list(self.conv1.parameters()) + list(self.conv2.parameters())

        return bn_params, other_params


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

    def get_parameters(self,):
        return list(self.bn.parameters()), list(self.conv.parameters())


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        dense_layers = []
        dense_layers += self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        dense_layers.append(Transition(num_planes, out_planes))
        num_planes = out_planes

        dense_layers += self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        dense_layers.append(Transition(num_planes, out_planes))
        num_planes = out_planes

        dense_layers += self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        dense_layers.append(Transition(num_planes, out_planes))
        num_planes = out_planes

        dense_layers += self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.dense_layers = nn.ModuleList(dense_layers)

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.initialize()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate

        return layers

    def forward(self, x):
        out = self.conv1(x)

        for layer in self.dense_layers:
            out = layer(out)

        out = F.relu(self.bn(out))
        out = out.mean(-1).mean(-1)
        out = self.linear(out)

        return out

    def initialize(self,):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def get_parameters(self,):
        bn_params, other_params = [], []

        other_params.extend(list(self.conv1.parameters()))

        for layer in self.dense_layers:
            bn, other = layer.get_parameters()
            bn_params.extend(bn)
            other_params.extend(other)

        bn_params.extend(list(self.bn.parameters()))
        other_params.extend(list(self.linear.parameters()))

        return bn_params, other_params


def load_pretrained_weights(model, model_name, n_class, pretrained):
    if pretrained == 'with_classifier':
        if n_class == 10:
            model_name = 'cifar10_{}'.format(model_name)
        elif n_class == 100:
            model_name = 'cifar100_{}'.format(model_name)
        else:
            model_name = ''

        state_dict = model_utils.get_cifar_pretrained_weights(model_name)

        if state_dict is not None:
            model.load_state_dict(state_dict)

    elif pretrained == 'without_classifier':
        model_name = 'cifar100_{}'.format(model_name)
        state_dict = model_utils.get_cifar_pretrained_weights(model_name)
        if state_dict is not None:
            cur_state_dict = model.state_dict()
            state_dict['linear.weight'] = cur_state_dict['linear.weight']
            state_dict['linear.bias'] = cur_state_dict['linear.bias']
            model.load_state_dict(state_dict)


class DenseNet121(DenseNet):
    def __init__(self, n_class, pretrained):
        super(DenseNet121, self).__init__(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=n_class)
        load_pretrained_weights(self, 'densenet121', n_class, pretrained)


class DenseNet169(DenseNet):
    def __init__(self, n_class, pretrained):
        super(DenseNet169, self).__init__(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=n_class)
        load_pretrained_weights(self, 'densenet169', n_class, pretrained)


class DenseNet201(DenseNet):
    def __init__(self, n_class, pretrained):
        super(DenseNet201, self).__init__(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_classes=n_class)
        load_pretrained_weights(self, 'densenet201', n_class, pretrained)


class DenseNet161(DenseNet):
    def __init__(self, n_class, pretrained):
        super(DenseNet161, self).__init__(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_classes=n_class)
        load_pretrained_weights(self, 'densenet161', n_class, pretrained)
