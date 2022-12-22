# Adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py

from typing import List, Type, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils import model_zoo
from torchvision import models


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(
        self,
        block: Type[Union[models.resnet.BasicBlock, models.resnet.Bottleneck]],
        layers: List[int],
        num_input_images: int = 1,
    ) -> None:
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(
    num_layers: int,
    pretrained: bool = False,
    num_input_images: int = 1,
) -> models.ResNet:
    """Constructs a resnet model with varying number of input images.

    :param num_layers: Specifies the ResNet model to be used
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param num_input_images: Specifies the number of input images. For PoseNet, >1.
    """
    if num_layers not in [18, 50]:
        raise ValueError('Resnet multi-image can only run with 18 or 50 layers.')

    block_type = {
        18: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck,
    }[num_layers]
    blocks = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3],
    }[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls[f'resnet{num_layers}'])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images,
                                           1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    AVAILABLE_RESNETS = {
        18: models.resnet18,
        34: models.resnet34,
        # 50: models.resnet50,
    }

    def __init__(
        self,
        num_layers: int,
        pretrained: bool,
        num_input_images: int = 1,
    ) -> None:
        """Constructs a ResNet model.

        :param num_layers: Specifies the ResNet model to be used
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param num_input_images: Specifies the number of input images. For PoseNet, >1.
        """
        super().__init__()
        self.num_ch_encoder = np.array([64, 64, 128, 256, 512])

        if num_layers not in self.AVAILABLE_RESNETS.keys():
            raise ValueError(f'Could not find a ResNet model with {num_layers} layers.')

        if num_input_images < 1:
            raise ValueError(f'Invalid value ({num_input_images}) for num_input_images.')
        if num_input_images == 1:
            self.resnet = self.AVAILABLE_RESNETS[num_layers](pretrained)
        else:
            self.resnet = resnet_multiimage_input(num_layers, pretrained, num_input_images)

        # From paper
        if num_layers > 34:
            self.num_ch_encoder[1:] *= 4

    def forward(self, x: Tensor) -> List[Tensor]:
        features = []
        x = (x - 0.45) / 0.225
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        features.append(self.resnet.relu(x))
        features.append(self.resnet.layer1(self.resnet.maxpool(features[-1])))
        features.append(self.resnet.layer2(features[-1]))
        features.append(self.resnet.layer3(features[-1]))
        features.append(self.resnet.layer4(features[-1]))
        return features
