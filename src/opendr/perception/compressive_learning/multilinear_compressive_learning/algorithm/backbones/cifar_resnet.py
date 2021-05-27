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

import torch.nn as nn
import torch.nn.functional as F
from . import model_utils


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.ModuleList([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)]
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            for layer in self.shortcut:
                x = layer(x)

        out += x
        out = F.relu(out)
        return out

    def get_parameters(self,):
        bn_params = list(self.bn1.parameters()) + list(self.bn2.parameters())
        other_params = list(self.conv1.parameters()) + list(self.conv2.parameters())

        if self.shortcut is not None:
            bn_params.extend(list(self.shortcut[1].parameters()))
            other_params.extend(list(self.shortcut[0].parameters()))

        return bn_params, other_params


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        return out

    def get_parameters(self,):
        bn_params = list(self.bn1.parameters()) + list(self.bn2.parameters())
        other_params = list(self.conv1.parameters()) + list(self.conv2.parameters()) +\
            list(self.shortcut.parameters())

        return bn_params, other_params


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.ModuleList([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)]
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.shortcut is not None:
            for layer in self.shortcut:
                x = layer(x)

        out += x
        out = F.relu(out)

        return out

    def get_parameters(self,):
        bn_params = list(self.bn1.parameters()) + list(self.bn2.parameters()) +\
            list(self.bn3.parameters())

        other_params = list(self.conv1.parameters()) + list(self.conv2.parameters()) +\
            list(self.conv3.parameters())

        if self.shortcut is not None:
            bn_params.extend(list(self.shortcut[1].parameters()))
            other_params.extend(list(self.shortcut[0].parameters()))

        return bn_params, other_params


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)

        residual_layers = []

        residual_layers += self._make_layer(block, 64, num_blocks[0], stride=1)
        residual_layers += self._make_layer(block, 128, num_blocks[1], stride=2)
        residual_layers += self._make_layer(block, 256, num_blocks[2], stride=2)
        residual_layers += self._make_layer(block, 512, num_blocks[3], stride=2)

        self.residual_layers = nn.ModuleList(residual_layers)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.initialize()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        for layer in self.residual_layers:
            out = layer(out)

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
        bn_params.extend(list(self.bn1.parameters()))

        for layer in self.residual_layers:
            bn, other = layer.get_parameters()
            bn_params.extend(bn)
            other_params.extend(other)

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


class ResNet18(ResNet):
    def __init__(self, n_class, pretrained):
        super(ResNet18, self).__init__(PreActBlock, [2, 2, 2, 2], num_classes=n_class)
        load_pretrained_weights(self, 'resnet18', n_class, pretrained)


class ResNet34(ResNet):
    def __init__(self, n_class, pretrained):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=n_class)
        load_pretrained_weights(self, 'resnet34', n_class, pretrained)


class ResNet50(ResNet):
    def __init__(self, n_class, pretrained):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=n_class)
        load_pretrained_weights(self, 'resnet50', n_class, pretrained)


class ResNet101(ResNet):
    def __init__(self, n_class, pretrained):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=n_class)
        load_pretrained_weights(self, 'resnet101', n_class, pretrained)


class ResNet152(ResNet):
    def __init__(self, n_class, pretrained):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes=n_class)
        load_pretrained_weights(self, 'resnet152', n_class, pretrained)
