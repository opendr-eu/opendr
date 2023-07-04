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


import json
import os
import sys
import requests
from logging import getLogger
from pathlib import Path
from re import match
from typing import Union
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm
from zipfile import ZipFile

import torch

from vosk import KaldiRecognizer, MODEL_PRE_URL, MODEL_LIST_URL, MODEL_DIRS
from vosk import Model as VoskModel

from opendr.engine.data import Timeseries
from opendr.engine.learners import Learner
from opendr.engine.target import VoskTranscription


logger = getLogger(__name__)


class VoskLearner(Learner):
    def __init__(
        self,
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        """
        The VoskLearner class extends the base Learner class and incorporates the
        functionality of the Vosk speech recognition library.

        Args:
            device: str
                The device to use for computations. Currently only supports CPU.
            sample_rate: int
                The sample rate to be used by the Vosk model.
        """

        super(VoskLearner, self).__init__()
        if device == "cuda":
            logger.warning(
                "The implementation does not support CUDA, using CPU instead."
            )
            device = "cpu"

        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self.rec = None

    def _load_model_weights(self, model_path) -> VoskModel:
        return VoskModel(model_path=model_path)

    def load(
        self,
        name: str = None,
        language: str = None,
        model_path=None,
        download_dir=None,
    ):
        self.model_name = name
        self.language = language
        self.download_dir = download_dir
        if model_path is None:
            model_path = self._get_model_path()

        self.model = self._load_model_weights(model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)

    def _get_model_path(self):
        if self.model_name is None:
            model_path = self._get_model_by_lang(self.language)
        else:
            model_path = self._get_model_by_name(self.model_name)
        return str(model_path)

    def _get_model_by_name(self, model_name: str):
        """
        Adpated from https://github.com/alphacep/vosk-api/blob/master/python/vosk/__init__.py#L72

        Args:
            model_name: str
                Full name of the Vosk model.
        """

        if self.download_dir is None:
            for directory in MODEL_DIRS:
                if directory is None or not Path(directory).exists():
                    continue
                model_file_list = os.listdir(directory)
                model_file = [model for model in model_file_list if model == model_name]
                if model_file != []:
                    return Path(directory, model_file[0])
        else:
            directory = self.download_dir

        response = requests.get(MODEL_LIST_URL, timeout=10)
        result_model = [
            model["name"] for model in response.json() if model["name"] == model_name
        ]
        if result_model == []:
            print("model name %s does not exist" % (model_name))
            sys.exit(1)
        else:
            self.download(Path(directory, result_model[0]))
            return Path(directory, result_model[0])

    def _get_model_by_lang(self, lang: str):
        """
        Adpated from https://github.com/alphacep/vosk-api/blob/master/python/vosk/__init__.py#L89

        Args:
            lang: str
                Language of the Vosk model. Vosk will decide he default model for this language.
        """

        if self.download_dir is None:
            for directory in MODEL_DIRS:
                if directory is None or not Path(directory).exists():
                    continue
                model_file_list = os.listdir(directory)
                model_file = [
                    model
                    for model in model_file_list
                    if match(r"vosk-model(-small)?-{}".format(lang), model)
                ]
                if model_file != []:
                    return Path(directory, model_file[0])
        else:
            directory = self.download_dir

        response = requests.get(MODEL_LIST_URL, timeout=10)
        result_model = [
            model["name"]
            for model in response.json()
            if model["lang"] == lang
            and model["type"] == "small"
            and model["obsolete"] == "false"
        ]
        if result_model == []:
            print("lang %s does not exist" % (lang))
            sys.exit(1)
        else:
            self.download(Path(directory, result_model[0]))
            return Path(directory, result_model[0])

    def download(self, model_name: str):
        """
        Adpated from https://github.com/alphacep/vosk-api/blob/master/python/vosk/__init__.py#L108

        Args:
            model_name: str
                Full name of the Vosk model.
        """

        if not (model_name.parent).exists():
            (model_name.parent).mkdir(parents=True)
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=(MODEL_PRE_URL + str(model_name.name) + ".zip").rsplit(
                "/", maxsplit=1
            )[-1],
        ) as t:
            reporthook = self.download_progress_hook(t)
            urlretrieve(
                MODEL_PRE_URL + str(model_name.name) + ".zip",
                str(model_name) + ".zip",
                reporthook=reporthook,
                data=None,
            )
            t.total = t.n
            with ZipFile(str(model_name) + ".zip", "r") as model_ref:
                model_ref.extractall(model_name.parent)
            Path(str(model_name) + ".zip").unlink()

    def download_progress_hook(self, t):
        """
        Adapted from https://github.com/alphacep/vosk-api/blob/master/python/vosk/__init__.py#L122
        """
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

        return update_to

    def infer(
        self, audio: Union[Timeseries, torch.Tensor, np.ndarray, bytes]
    ) -> VoskTranscription:
        """
        Run inference on an audio sample.

        Args:
            audio (Union[Timeseries, np.ndarray, torch.Tensor, bytes]):
                The audio sample as a Timeseries, torch.Tensor, or np.ndarray or bytes.

        Returns:
            VoskTranscription: Transcription results with side informmation.

        Raises:
            TypeError: If the input batch is not a Timeseries, torch.Tensor, np.ndarray or str.
        """

        if isinstance(audio, (Timeseries, torch.Tensor)):
            data = audio.numpy().reshape(-1)
        elif isinstance(audio, (bytes, np.ndarray)):
            data = audio
        else:
            raise TypeError(
                "batch must be a timeseries, bytes, torch.tensor or np.ndarray"
            )

        byte_data = self.preprocess(data)
        accept_waveform = self.rec.AcceptWaveform(byte_data)
        if accept_waveform:
            output = self.rec.Result()
            text = json.loads(output)["text"]
        else:
            output = self.rec.PartialResult()
            text = json.loads(output)["partial"]

        return VoskTranscription(text=text, accept_waveform=accept_waveform)

    def preprocess(self, data: Union[np.ndarray, bytes]) -> bytes:
        """
        Preprocess audio data.

        Args:
            data (Union[np.ndarray, bytes]):
                pad or trim the data to a desired length for Whisper and compute log mel spectrogram.

        Returns:
            bytes: Audio data in bytes.
        """
        if isinstance(data, bytes):
            return data

        # Convert the array to int16, as vosk expects 16-bit integer data.
        data = (data * np.iinfo(np.int16).max).astype(np.int16).tobytes()

        return data

    def save(self):
        return

    def reset(self):
        return

    def fit(self):
        return

    def eval(self):
        return

    def optimize(self):
        return

    def _save_onnx(self):
        return
