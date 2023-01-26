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

import torch
import pytorch_lightning as pl


class _LightningModuleWithCrossEntropy(pl.LightningModule):
    def __init__(self, module):
        pl.LightningModule.__init__(self)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.module(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("train/loss", loss)
        self.log("train/acc", _accuracy(z, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("val/loss", loss)
        self.log("val/acc", _accuracy(z, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("test/loss", loss)
        self.log("test/acc", _accuracy(z, y))
        return loss


def _accuracy(x, y):
    return torch.sum(x.argmax(dim=1) == y) / len(y)
