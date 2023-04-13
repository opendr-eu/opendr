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


from typing import Union, Iterable, Optional, List, Dict
import os
import io
import hashlib
from logging import getLogger
import warnings
import urllib

import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch

import whisper
from whisper import _MODELS as MODELS_URL
from whisper.model import ModelDimensions, Whisper

from opendr.engine.data import Timeseries
from opendr.engine.learners import Learner
from opendr.engine.datasets import Dataset


logger = getLogger(__name__)

_MODEL_NAMES = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']

class WhisperLearner(Learner):
    def __init__(
            self,
            fp16: bool = True,
        ):
        self.sample_rate = 16000
        self.fp16 = fp16 


    def _load_model_hparams(self):
        return

    def _load_model_weights(self, load_path: Union[str, Path], device: Optional[Union[str, torch.device]] = None, in_memory: bool = False):
        """Load pretrained weight using Whisper api: whisper.load_model

        :molde_name: name of pretrained model
        :returns: TODO

        """
        
        if os.path.isfile(load_path):
            checkpoint_file = open(load_path, "rb").read() if in_memory else load_path
            alignment_heads = None
        else:
            raise RuntimeError(
                f"Model {load_path} not found; available models = {whisper.available_models()}"
            )

        with (
            io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
        ) as fp:
            checkpoint = torch.load(fp, map_location=device)
        del checkpoint_file

        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])

        if alignment_heads is not None:
            model.set_alignment_heads(alignment_heads)

        return model.to(device)
        

    def init_model(self):
        return

    def save(self, path: Union[str, Path]):
        """Save model weights to path

        :path: Directory in which to save model weights. 
        :returns: TODO

        """
        pass

    def load(self, load_path: Union[str, Path] = None, model_name: str = None, device: Optional[Union[str, torch.device]] = None, download_root: Union[str, Path] = "./", in_memory: bool = False):
        """Load model.

        :path: TODO
        :returns: TODO

        """
        assert load_path is not None or model_name is not None, "Either load_path or model_name must be provided"
        
        if load_path is None:
            self.download(model_name=model_name, path=download_root)

            url = MODELS_URL[model_name]
            load_path = os.path.join(download_root, os.path.basename(url))

        self.model = self._load_model_weights(
            load_path=load_path,
            device=device,
            in_memory=in_memory,
        )


    @staticmethod
    def download(model_name: str, path: Union[str, Path]):

        url = MODELS_URL[model_name]
        os.makedirs(path, exist_ok=True)
        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(path, os.path.basename(url))

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        if os.path.isfile(download_target):

            with open(download_target, "rb") as f:
                model_bytes = f.read()
            if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
                pass
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
            )


    def reset(self):
        return

    def fit(self):
        return

    def eval(self, dataset: Dataset, steps: int = None):
        """Evaluate the model on the dataset.

        :steps: TODO
        :returns: TODO
        :dataset: TODO

        """
        pass

    def infer(self, batch: Union[Timeseries, np.array, torch.Tensor], temperature: float = 0.0, device: Union[str, torch.device] = "cpu", **decode_options) -> Dict:
        """Run inference on one sample of audio

        :batch: 1-D array or 2-D array with last dimension is 1.
        :returns: TODO

        """
        if isinstance(batch, Timeseries):
            data = batch.numpy().reshape(-1)
        elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
            data = batch
        else:
            raise TypeError("batch must be a Timeseries, torch.Tensor or np.ndarray")

        data = whisper.pad_or_trim(data)       
        mel = whisper.log_mel_spectrogram(data, device=device)

        decode_options["language"] = "en"
        decode_options["without_timestamps"] = False
        decode_options["fp16"] = self.fp16
        options = whisper.DecodingOptions(**decode_options, temperature=temperature)
        output = whisper.decode(model=self.model, mel=mel, options=options)

        return output

    def optimize(self):
        return

    def _save_onnx(self):
        return

