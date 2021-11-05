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
from . import model_utils


class AllCNN(nn.Module):
    def __init__(self, n_class, pretrained):
        super(AllCNN, self).__init__()

        self.block1 = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block2 = nn.ModuleList([
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block3 = nn.ModuleList([
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        ])

        self.classifier = nn.ModuleList([
            nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        ])

        if pretrained == 'with_classifier':
            if n_class == 10:
                model_name = 'cifar10_allcnn'
            elif n_class == 100:
                model_name = 'cifar100_allcnn'
            else:
                model_name = ''

            state_dict = model_utils.get_cifar_pretrained_weights(model_name)
            if state_dict is not None:
                self.load_state_dict(state_dict)
        elif pretrained == 'without_classifier':
            model_name = 'cifar100_allcnn'
            state_dict = model_utils.get_cifar_pretrained_weights(model_name)
            cur_state_dict = self.state_dict()
            for key in cur_state_dict.keys():
                if key.startswith('classifier'):
                    state_dict[key] = cur_state_dict[key]

    def forward(self, x):
        for layer in self.block1:
            x = layer(x)
        for layer in self.block2:
            x = layer(x)
        for layer in self.block3:
            x = layer(x)
        for layer in self.classifier:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return x

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
        bn_params = list(self.block1[1].parameters()) +\
            list(self.block1[4].parameters()) +\
            list(self.block1[7].parameters()) +\
            list(self.block2[1].parameters()) +\
            list(self.block2[4].parameters()) +\
            list(self.block2[7].parameters()) +\
            list(self.block3[1].parameters()) +\
            list(self.block3[4].parameters()) +\
            list(self.classifier[1].parameters())

        other_params = list(self.block1[0].parameters()) +\
            list(self.block1[3].parameters()) +\
            list(self.block1[6].parameters()) +\
            list(self.block2[0].parameters()) +\
            list(self.block2[3].parameters()) +\
            list(self.block2[6].parameters()) +\
            list(self.block3[0].parameters()) +\
            list(self.block3[3].parameters()) +\
            list(self.classifier[0].parameters())

        return bn_params, other_params
