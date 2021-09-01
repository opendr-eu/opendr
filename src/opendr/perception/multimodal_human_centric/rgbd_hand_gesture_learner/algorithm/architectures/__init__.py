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

from .mobilenet import mobilenet_v2, model_urls as mobilenet_urls
from .vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
    model_urls as vgg_urls
)
from .mnasnet import (
    mnasnet0_5,
    mnasnet0_75,
    mnasnet1_0,
    mnasnet1_3,
    model_urls as mnasnet_urls
)
from .densenet import (
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    model_urls as densenet_urls
)
from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
    model_urls as resnet_urls
)

architectures = {'mobilenet_v2': mobilenet_v2,
                 'vgg11': vgg11,
                 'vgg11_bn': vgg11_bn,
                 'vgg13': vgg13,
                 'vgg13_bn': vgg13_bn,
                 'vgg16': vgg16,
                 'vgg16_bn': vgg16_bn,
                 'vgg19': vgg19,
                 'vgg19_bn': vgg19_bn,
                 'mnasnet0_5': mnasnet0_5,
                 'mnasnet0_75': mnasnet0_75,
                 'mnasnet1_0': mnasnet1_0,
                 'mnasnet1_3': mnasnet1_3,
                 'densenet121': densenet121,
                 'densenet161': densenet161,
                 'densenet169': densenet169,
                 'densenet201': densenet201,
                 'resnet18': resnet18,
                 'resnet34': resnet34,
                 'resnet50': resnet50,
                 'resnet101': resnet101,
                 'resnet152': resnet152,
                 'resnext50_32x4d': resnext50_32x4d,
                 'resnext101_32x8d': resnext101_32x8d,
                 'wide_resnet50_2': wide_resnet50_2,
                 'wide_resnet101_2': wide_resnet101_2}


def get_builtin_architectures():
    return list(architectures.keys())


def get_pretrained_architectures():
    return list(vgg_urls.keys()) +\
        list(mobilenet_urls.keys()) +\
        list(mnasnet_urls.keys()) +\
        list(densenet_urls.keys()) +\
        list(resnet_urls.keys())
