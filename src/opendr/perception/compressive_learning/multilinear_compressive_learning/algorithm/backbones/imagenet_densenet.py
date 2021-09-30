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
import torchvision.models.densenet as densenet_models
import re

getters = {'densenet121': densenet_models.densenet121,
           'densenet161': densenet_models.densenet161,
           'densenet169': densenet_models.densenet169,
           'densenet201': densenet_models.densenet201,
           }


def extract_parameters_from_dense_block(block):
    bn_params, other_params = [], []
    for _, layer in block.items():
        bn_params.extend(list(layer.norm1.parameters()))
        bn_params.extend(list(layer.norm2.parameters()))

        other_params.extend(list(layer.conv1.parameters()))
        other_params.extend(list(layer.conv2.parameters()))

    return bn_params, other_params


def extract_parameters_from_transition_block(block):
    bn_params, other_params = [], []
    bn_params.extend(list(block.norm.parameters()))
    other_params.extend(list(block.conv.parameters()))

    return bn_params, other_params


class DenseNet(nn.Module):
    def __init__(self, model_name, n_class, pretrained):
        super(DenseNet, self).__init__()

        assert model_name in getters.keys()
        assert pretrained in ['', 'with_classifier', 'without_classifier']

        self.body = getters[model_name](pretrained=False, num_classes=n_class)

        if pretrained != '':
            # get pretrained weights
            state_dict = densenet_models.load_state_dict_from_url(densenet_models.model_urls[model_name],
                                                                  progress=False)

            # fix naming issue
            # as in https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            # remove the last classifier layer from pretrained weights
            if pretrained == 'without_classifier':
                cur_state_dict = self.body.state_dict()
                state_dict['classifier.weight'] = cur_state_dict['classifier.weight']
                state_dict['classifier.bias'] = cur_state_dict['classifier.bias']
            else:
                assert n_class == 1000,\
                    'For option pretrained="with_imagenet_classifier", ' +\
                    'the number of classes must be 1000. Provided value: {}'.format(n_class)

            self.body.load_state_dict(state_dict)

    def forward(self, x):
        return self.body(x)

    def get_parameters(self,):
        bn_params, other_params = [], []

        for layer in self.body.features:
            if isinstance(layer, densenet_models._DenseBlock):
                bn, other = extract_parameters_from_dense_block(layer)
                bn_params.extend(bn)
                other_params.extend(other)
            elif isinstance(layer, densenet_models._Transition):
                bn, other = extract_parameters_from_transition_block(layer)
                bn_params.extend(bn)
                other_params.extend(other)
            elif isinstance(layer, nn.BatchNorm2d):
                bn_params.extend(list(layer.parameters()))
            else:
                other_params.extend(list(layer.parameters()))

        other_params.extend(list(self.body.classifier.parameters()))

        return bn_params, other_params


class DenseNet121(DenseNet):
    def __init__(self, n_class, pretrained=False):
        super(DenseNet121, self).__init__('densenet121', n_class, pretrained)


class DenseNet161(DenseNet):
    def __init__(self, n_class, pretrained=False):
        super(DenseNet161, self).__init__('densenet161', n_class, pretrained)


class DenseNet169(DenseNet):
    def __init__(self, n_class, pretrained=False):
        super(DenseNet169, self).__init__('densenet169', n_class, pretrained)


class DenseNet201(DenseNet):
    def __init__(self, n_class, pretrained=False):
        super(DenseNet201, self).__init__('densenet201', n_class, pretrained)
