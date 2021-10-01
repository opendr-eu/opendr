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

from . import backbones
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_builtin_backbones():
    return list(backbones.models.keys())


class MultilinearMapping(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MultilinearMapping, self).__init__()

        assert len(input_shape) == 3,\
            'Input shape must be a 3-tuple containing (#row, #col, #channel). Received "{}"'.format(input_shape)

        assert len(output_shape) == 3,\
            'Output shape must be a 3-tuple containing (#row, #col, #channel). Received "{}"'.format(output_shape)

        # reorder input shape and compressed shape to (#channel, #row, #col)
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        output_shape = (output_shape[2], output_shape[0], output_shape[1])

        # parameters to map the channel dimension
        use_first_mode = False if (input_shape[0] == 1 and output_shape[0] == 1) else True
        if use_first_mode:
            self.W1 = nn.Parameter(data=torch.Tensor(output_shape[0], input_shape[0]),
                                   requires_grad=True)
        else:
            self.W1 = None

        # parameters to map the spatial dimension (rows)
        self.W2 = nn.Parameter(data=torch.Tensor(output_shape[1], input_shape[1]),
                               requires_grad=True)

        # parameters to map the spatial dimension (cols)
        self.W3 = nn.Parameter(data=torch.Tensor(output_shape[2], input_shape[2]),
                               requires_grad=True)

        # initialization
        nn.init.kaiming_uniform_(self.W3)
        nn.init.kaiming_uniform_(self.W2)
        if self.W1 is not None:
            nn.init.kaiming_uniform_(self.W1)

    def forward(self, x):
        x = self.nmodeproduct(x, self.W2, 2)
        x = self.nmodeproduct(x, self.W3, 3)
        if self.W1 is not None:
            x = self.nmodeproduct(x, self.W1, 1)

        return x

    def nmodeproduct(self, x, W, mode):
        if mode == 1:
            y = torch.transpose(x, 1, 3)
            y = F.linear(y, W)
            y = torch.transpose(y, 1, 3)
        elif mode == 2:
            y = torch.transpose(x, 2, 3)
            y = F.linear(y, W)
            y = torch.transpose(y, 2, 3)
        else:
            y = F.linear(x, W)

        return y


class SenseSynth(nn.Module):
    def __init__(self, input_shape, compressed_shape):
        super(SenseSynth, self).__init__()

        # sensing module
        self.sensing_module = MultilinearMapping(input_shape, compressed_shape)

        # synthesis module
        self.synthesis_module = MultilinearMapping(compressed_shape, input_shape)

    def forward(self, x):
        x = self.sensing_module(x)
        x = self.synthesis_module(x)
        return x


class CompressiveLearner(nn.Module):
    def __init__(self, input_shape, compressed_shape, n_class, backbone, pretrained_backbone):

        super(CompressiveLearner, self).__init__()

        if isinstance(backbone, str):
            # built-in backbone given as a string
            assert backbone in backbones.models.keys(),\
                'Given `backbone` parameter "{}" is not supported. '.format(backbone) +\
                'Supported `backbone` includes:\n' +\
                '\n'.join(['"{}"'.format(name) for name in backbones.models.keys()])

            self.backbone = backbones.models[backbone](n_class, pretrained_backbone)

        else:
            assert isinstance(backbone, torch.nn.Module),\
                '`backbone` parameter must be a string indicating built-in backbone ' +\
                'or an instance of torch.nn.Module\n' +\
                'received backbone of type: {}'.format(type(backbone))

            self.backbone = backbone

        self.sense_synth_module = SenseSynth(input_shape, compressed_shape)

    def forward(self, x):
        x = self.sense_synth_module(x)
        x = self.backbone(x)
        return x

    def infer_from_measurement(self, x):
        x = self.sense_synth_module.synthesis_module(x)
        x = self.backbone(x)
        return x

    def get_parameters(self,):
        bn_params, other_params = [], []
        if hasattr(self.backbone, 'get_parameters') and callable(self.backbone.get_parameters):
            backbone_bn_params, backbone_other_params = self.backbone.get_parameters()
            bn_params.extend(backbone_bn_params)
            other_params.extend(backbone_other_params)
            other_params.extend(list(self.sense_synth_module.parameters()))
        else:
            other_params.extend(list(self.sense_synth_module.parameters()))
            other_params.extend(list(self.backbone.parameters()))

        return bn_params, other_params
