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

import torch.nn as nn
import torchvision.models.resnet as resnet_models

getters = {'resnet18': resnet_models.resnet18,
           'resnet34': resnet_models.resnet34,
           'resnet50': resnet_models.resnet50,
           'resnet101': resnet_models.resnet101,
           'resnet152': resnet_models.resnet152,
           'resnext50_32x4d': resnet_models.resnext50_32x4d,
           'resnext101_32x8d': resnet_models.resnext101_32x8d,
           'wide_resnet50_2': resnet_models.wide_resnet50_2,
           'wide_resnet101_2': resnet_models.wide_resnet101_2,
           }


class ResNet(nn.Module):
    def __init__(self, model_name, n_class, pretrained):
        super(ResNet, self).__init__()

        assert model_name in getters.keys()
        assert pretrained in ['', 'with_classifier', 'without_classifier']

        self.body = getters[model_name](pretrained=False, num_classes=n_class)

        if pretrained != '':
            # get pretrained weights
            state_dict = resnet_models.load_state_dict_from_url(resnet_models.model_urls[model_name], progress=False)

            # remove the last classifier layer from pretrained weights
            if pretrained == 'without_classifier':
                cur_state_dict = self.body.state_dict()
                state_dict['fc.weight'] = cur_state_dict['fc.weight']
                state_dict['fc.bias'] = cur_state_dict['fc.bias']
            else:
                assert n_class == 1000,\
                    'For option pretrained="with_imagenet_classifier", ' +\
                    'the number of classes must be 1000. Provided value: {}'.format(n_class)

            self.body.load_state_dict(state_dict)

    def forward(self, x):
        return self.body(x)

    def get_parameters(self,):
        bn_params, other_params = [], []

        other_params.extend(list(self.body.conv1.parameters()))
        bn_params.extend(list(self.body.bn1.parameters()))

        for block in [self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4]:
            for layer in block:
                if isinstance(layer, nn.BatchNorm2d):
                    bn_params.extend(list(layer.parameters()))
                else:
                    other_params.extend(list(layer.parameters()))

        other_params.extend(list(self.body.fc.parameters()))

        return bn_params, other_params


class ResNet18(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNet18, self).__init__('resnet18', n_class, pretrained)


class ResNet34(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNet34, self).__init__('resnet34', n_class, pretrained)


class ResNet50(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNet50, self).__init__('resnet50', n_class, pretrained)


class ResNet101(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNet101, self).__init__('resnet101', n_class, pretrained)


class ResNet152(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNet152, self).__init__('resnet152', n_class, pretrained)


class ResNext50_32x4d(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNext50_32x4d, self).__init__('resnext50_32x4d', n_class, pretrained)


class ResNext101_32x8d(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(ResNext101_32x8d, self).__init__('resnext101_32x8d', n_class, pretrained)


class WideResNet50_2(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(WideResNet50_2, self).__init__('wide_resnet50_2', n_class, pretrained)


class WideResNet101_2(ResNet):
    def __init__(self, n_class, pretrained=False):
        super(WideResNet101_2, self).__init__('wide_resnet101_2', n_class, pretrained)
