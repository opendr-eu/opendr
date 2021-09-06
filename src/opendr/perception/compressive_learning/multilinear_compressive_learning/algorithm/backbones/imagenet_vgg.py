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
import torchvision.models.vgg as vgg_models

getters = {'vgg11': vgg_models.vgg11,
           'vgg11_bn': vgg_models.vgg11_bn,
           'vgg13': vgg_models.vgg13,
           'vgg13_bn': vgg_models.vgg13_bn,
           'vgg16': vgg_models.vgg16,
           'vgg16_bn': vgg_models.vgg16_bn,
           'vgg19': vgg_models.vgg19,
           'vgg19_bn': vgg_models.vgg19_bn}


class VGG(nn.Module):
    def __init__(self, model_name, n_class, pretrained):
        super(VGG, self).__init__()

        assert model_name in getters.keys()
        assert pretrained in ['', 'with_classifier', 'without_classifier']

        self.body = getters[model_name](pretrained=False, num_classes=n_class)

        if pretrained != '':
            # get pretrained weights
            state_dict = vgg_models.load_state_dict_from_url(vgg_models.model_urls[model_name], progress=False)

            # remove the last classifier layer from pretrained weights
            if pretrained == 'without_classifier':
                cur_state_dict = self.body.state_dict()
                state_dict['classifier.6.weight'] = cur_state_dict['classifier.6.weight']
                state_dict['classifier.6.bias'] = cur_state_dict['classifier.6.bias']
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
            if isinstance(layer, nn.BatchNorm2d):
                bn_params.extend(list(layer.parameters()))
            else:
                other_params.extend(list(layer.parameters()))

        other_params.extend(list(self.body.classifier.parameters()))

        return bn_params, other_params


class VGG11(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG11, self).__init__('vgg11', n_class, pretrained)


class VGG11_BN(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG11_BN, self).__init__('vgg11_bn', n_class, pretrained)


class VGG13(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG13, self).__init__('vgg13', n_class, pretrained)


class VGG13_BN(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG13_BN, self).__init__('vgg13_bn', n_class, pretrained)


class VGG16(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG16, self).__init__('vgg16', n_class, pretrained)


class VGG16_BN(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG16_BN, self).__init__('vgg16_bn', n_class, pretrained)


class VGG19(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG19, self).__init__('vgg19', n_class, pretrained)


class VGG19_BN(VGG):
    def __init__(self, n_class, pretrained=False):
        super(VGG19_BN, self).__init__('vgg19_bn', n_class, pretrained)
