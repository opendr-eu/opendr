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

import torch


class DataWrapper:
    def __init__(self, opendr_dataset):
        self.dataset = opendr_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset.__getitem__(i)
        # change from rows x cols x channels to channels x rows x cols
        x = x.convert("channels_first", "rgb")
        return torch.from_numpy(x).float(), torch.tensor([y.data, ]).long()
