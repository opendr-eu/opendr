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

# cifar
from .cifar_allcnn import AllCNN as CifarAllCNN
from .cifar_densenet import DenseNet121 as CifarDenseNet121
from .cifar_densenet import DenseNet169 as CifarDenseNet169
from .cifar_densenet import DenseNet201 as CifarDenseNet201
from .cifar_densenet import DenseNet161 as CifarDenseNet161
from .cifar_resnet import ResNet18 as CifarResNet18
from .cifar_resnet import ResNet34 as CifarResNet34
from .cifar_resnet import ResNet50 as CifarResNet50
from .cifar_resnet import ResNet101 as CifarResNet101
from .cifar_resnet import ResNet152 as CifarResNet152
from .cifar_vgg import VGG11 as CifarVGG11
from .cifar_vgg import VGG13 as CifarVGG13
from .cifar_vgg import VGG16 as CifarVGG16
from .cifar_vgg import VGG19 as CifarVGG19

# imagenet
from .imagenet_vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN
from .imagenet_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .imagenet_resnet import ResNext50_32x4d, ResNext101_32x8d, WideResNet50_2, WideResNet101_2
from .imagenet_densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201

models = {'cifar_allcnn': CifarAllCNN,
          'cifar_densenet121': CifarDenseNet121,
          'cifar_densenet169': CifarDenseNet169,
          'cifar_densenet201': CifarDenseNet201,
          'cifar_densenet161': CifarDenseNet161,
          'cifar_resnet18': CifarResNet18,
          'cifar_resnet34': CifarResNet34,
          'cifar_resnet50': CifarResNet50,
          'cifar_resnet101': CifarResNet101,
          'cifar_resnet152': CifarResNet152,
          'cifar_vgg11': CifarVGG11,
          'cifar_vgg13': CifarVGG13,
          'cifar_vgg16': CifarVGG16,
          'cifar_vgg19': CifarVGG19,
          'imagenet_vgg11': VGG11,
          'imagenet_vgg11_bn': VGG11_BN,
          'imagenet_vgg13': VGG13,
          'imagenet_vgg13_bn': VGG13_BN,
          'imagenet_vgg16': VGG16,
          'imagenet_vgg16_bn': VGG16_BN,
          'imagenet_vgg19': VGG19,
          'imagenet_vgg19_bn': VGG19_BN,
          'imagenet_resnet18': ResNet18,
          'imagenet_resnet34': ResNet34,
          'imagenet_resnet50': ResNet50,
          'imagenet_resnet101': ResNet101,
          'imagenet_resnet152': ResNet152,
          'imagenet_resnext50_32x4d': ResNext50_32x4d,
          'imagenet_resnext101_32x8d': ResNext101_32x8d,
          'imagenet_wide_resnet50_2': WideResNet50_2,
          'imagenet_wide_resnet101_2': WideResNet101_2,
          'imagenet_densenet121': DenseNet121,
          'imagenet_densenet161': DenseNet161,
          'imagenet_densenet169': DenseNet169,
          'imagenet_densenet201': DenseNet201,
          }
