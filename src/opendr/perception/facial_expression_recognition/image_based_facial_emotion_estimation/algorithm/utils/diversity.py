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

"""
Implementation of diversity calculation between features extracted by different branches of  ESR
"""

import torch
import torch.nn as nn


class BranchDiversity(nn.Module):
    def __init__(self, ):
        super(BranchDiversity, self).__init__()
        self.direct_div = 0
        self.det_div = 0
        self.logdet_div = 0

    def forward(self, x, type='spatial'):

        num_branches = x.size(0)
        gamma = 10
        snm = torch.zeros((num_branches, num_branches))

        # Spatial attnention diversity
        if type == 'spatial':  # num_branch x batch_size x 6 x 6
            # diversity between spatial attention heads
            for i in range(num_branches):
                for j in range(num_branches):
                    if i != j:
                        diff = torch.exp(-1 * gamma * torch.sum(torch.square(x[i, :, :, :] - x[j, :, :, :]), (1, 2)))
                        # size: batch_size
                        diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # size: 1
                        snm[i, j] = diff
            self.direct_div = torch.sum(snm)
            self.det_div = -1 * torch.det(snm)
            self.logdet_div = -1 * torch.logdet(snm)

        # Channel attn diversity
        elif type == 'channel':  # num_branch x batch_size x 512
            # diversity between channels of attention heads
            for i in range(num_branches):
                for j in range(num_branches):
                    if i != j:
                        diff = torch.exp(
                            -1 * gamma * torch.sum(torch.square(x[i, :, :] - x[j, :, :]), 1))  # size: batch_size
                        diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # size: 1
                        snm[i, j] = diff
            self.direct_div = torch.sum(snm)
            self.det_div = -1 * torch.det(snm)
            self.logdet_div = -1 * torch.logdet(snm)

        return self
