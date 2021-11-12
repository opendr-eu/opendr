# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
# Modifications Copyright 2021 - present, OpenDR European Project

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

"""
Backbone modules.
"""

import torch
from torch import nn

try:
    # For older versions of torchvision
    from torchvision.models.mobilenet import InvertedResidual, mobilenet_v2
except ImportError:
    from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from opendr.perception.object_detection_2d.detr.algorithm.util.misc import NestedTensor
from opendr.perception.object_detection_2d.detr.algorithm.models.position_encoding import build_position_encoding

import torch.nn.functional as F


class BackboneBase(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            extra_blocks: nn.Module,
            train_backbone: bool,
            return_layers_backbone: dict,
            return_layers_extra_blocks: dict,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers_backbone)
        print("MobileNet activated")

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class MobileNetWithExtraBlocks(BackboneBase):
    """MobileNet backbone with extra blocks."""

    def __init__(
            self,
            train_backbone: bool,
    ):
        backbone = mobilenet_v2(pretrained=True).features
        return_layers_backbone = {"13": "0", "18": "1"}

        num_channels = 1280
        hidden_dims = [512, 256, 256, 64]
        expand_ratios = [0.2, 0.25, 0.5, 0.25]
        strides = [2, 2, 2, 2]
        extra_blocks = ExtraBlocks(num_channels, hidden_dims, expand_ratios, strides)
        return_layers_extra_blocks = {"0": "2", "1": "3", "2": "4", "3": "5"}

        super().__init__(
            backbone,
            extra_blocks,
            train_backbone,
            return_layers_backbone,
            return_layers_extra_blocks,
        )


class ExtraBlocks(nn.Sequential):
    def __init__(self, in_channels, hidden_dims, expand_ratios, strides):
        extra_blocks = []

        for i in range(len(expand_ratios)):
            input_dim = hidden_dims[i - 1] if i > 0 else in_channels
            extra_blocks.append(InvertedResidual(input_dim, hidden_dims[i], strides[i], expand_ratios[i]))

        super().__init__(*extra_blocks)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = MobileNetWithExtraBlocks(train_backbone)
    model = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels
    model.num_channels = 1280
    return model
