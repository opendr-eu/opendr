# Copyright 2020-2024 OpenDR European Project
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


from typing import Dict, List, Tuple, Type, Union

import torch as th
from stable_baselines3.common.utils import get_device
from torch import nn


class MlpExtractor_Aux(nn.Module):
    """
    analogous to the below MlpExtractor
    """

    def __init__(
            self,
            feature_dim: int,
            net_arch: List[Union[int, Dict[str, List[int]]]],
            activation_fn: Type[nn.Module],
            device="auto",
    ):
        super(MlpExtractor_Aux, self).__init__()
        device = get_device(device)
        last_layer_dim_aux = feature_dim
        aux_net = []

        # Simple auxilliary Network
        aux_net.append(nn.Linear(last_layer_dim_aux, 256))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(256, 128))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(128, 64))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(64, 32))
        aux_net.append(activation_fn())
        last_layer_dim_aux = 32

        # Save dim, used to create the distributions
        self.latent_dim_aux = last_layer_dim_aux

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module

        self.aux_net = nn.Sequential(*aux_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.aux_net(features)
