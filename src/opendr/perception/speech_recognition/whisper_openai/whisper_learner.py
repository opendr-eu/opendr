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


from typing import Union, Iterable
from pathlib import Path
from logging import getLogger

import torch

import whisper

from opendr.engine.learners import Learner
from opendr.engine.datasets import Dataset


logger = getLogger(__name__)

_MODEL_NAMES = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']

class WhisperLearner(Learner):
    def __init__(self):
        return

    def _load_model_hparams(self):
        return

    def _load_model_weights(self, molde_name: str):
        """Load pretrained weight using Whisper api: whisper.load_model

        :molde_name: name of pretrained model
        :returns: TODO

        """
        pass

    def init_model(self):
        return

    def save(self, path: Union[str, Path]):
        """Save model weights to path

        :path: Directory in which to save model weights. 
        :returns: TODO

        """
        pass

    def load(self, path: Union[str, Path]):
        """Load model.

        :path: TODO
        :returns: TODO

        """
        pass

    @staticmethod
    def download(model_names: Iterable[str] = _MODEL_NAMES):
        return

    def reset(self):
        return

    def fit(self):
        return

    def eval(self, dataset: Dataset, steps: int = None):
        """Evaluate the model on the dataset.

        :dataset: TODO
        :steps: TODO
        :returns: TODO

        """
        pass

    def infer(self, batch: Union[torch.Tensor]):
        """Run inference on a batch of data

        :batch: TODO
        :returns: TODO

        """

    def optimize(self):
        return

    def _save_onnx(self):
        return

