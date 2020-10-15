# Copyright 1996-2020 OpenDR European Project
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

from torch import nn


def get_parameters(model, predicate):
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)
