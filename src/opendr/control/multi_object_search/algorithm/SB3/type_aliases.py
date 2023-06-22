# Copyright 2020-2023 OpenDR European Project
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


from typing import Dict, Union
import torch as th

TensorDict = Dict[Union[str, int], th.Tensor]


class DictRolloutBufferSamples():
    def __init__(self, observations, actions, old_values, old_log_prob, advantages, returns, aux_angle, aux_angle_gt):
        self.observations = observations
        self.actions = actions
        self.old_values = old_values
        self.old_log_prob = old_log_prob
        self.advantages = advantages
        self.returns = returns
        self.aux_angle = aux_angle
        self.aux_angle_gt = aux_angle_gt
