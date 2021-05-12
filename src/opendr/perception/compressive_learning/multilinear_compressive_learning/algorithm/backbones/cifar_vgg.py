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
from . import model_utils

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, n_class):
        super(VGG, self).__init__()
        self.features = nn.ModuleList(self._make_layers(cfg[vgg_name]))
        self.classifier = nn.Linear(512, n_class)
        self.initialize()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return layers

    def initialize(self,):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def get_parameters(self,):
        bn_params, other_params = [], []
        for layer in self.features:
            if isinstance(layer, nn.BatchNorm2d):
                bn_params.extend(list(layer.parameters()))
            else:
                other_params.extend(list(layer.parameters()))

        other_params.extend(list(self.classifier.parameters()))

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
            state_dict['classifier.weight'] = cur_state_dict['classifier.weight']
            state_dict['classifier.bias'] = cur_state_dict['classifier.bias']
            model.load_state_dict(state_dict)


class VGG11(VGG):
    def __init__(self, n_class, pretrained):
        super(VGG11, self).__init__('VGG11', n_class)
        load_pretrained_weights(self, 'vgg11', n_class, pretrained)


class VGG13(VGG):
    def __init__(self, n_class, pretrained):
        super(VGG13, self).__init__('VGG13', n_class)
        load_pretrained_weights(self, 'vgg13', n_class, pretrained)


class VGG16(VGG):
    def __init__(self, n_class, pretrained):
        super(VGG16, self).__init__('VGG16', n_class)
        load_pretrained_weights(self, 'vgg16', n_class, pretrained)


class VGG19(VGG):
    def __init__(self, n_class, pretrained):
        super(VGG19, self).__init__('VGG19', n_class)
        load_pretrained_weights(self, 'vgg19', n_class, pretrained)
