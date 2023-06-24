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
from dataclasses import asdict
import os
import io
import hashlib
from logging import getLogger
import warnings
import urllib

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import whisper
from whisper import DecodingResult
from whisper import _MODELS as MODELS_URL, _ALIGNMENT_HEADS as ALIGNMENT_HEADS
from whisper.model import ModelDimensions, Whisper
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

from opendr.engine.data import Timeseries
from opendr.engine.target import WhisperTranscription
from opendr.engine.learners import Learner


logger = getLogger(__name__)


class WhisperLearner(Learner):
    def __init__(
        self,
        language: Optional[str] = "en",
        device: str = "cuda",
        logprob_threshold: float = -0.8,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        temperature: float = 0.0,
        sample_len: Optional[int] = None,
        best_of: Optional[int] = 5,
        beam_size: Optional[int] = 5,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        prompt: Optional[Union[str, List[int]]] = None,
        prefix: Optional[Union[str, List[int]]] = None,
        suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1",
        suppress_blank: bool = True,
        without_timestamps: bool = False,
        max_initial_timestamp: Optional[float] = 1.0,
        fp16: bool = True,
    ):
        """
        Initialize a new instance of the WhisperLearner class, a subclass of the Learner class.

        Args:
            model_name (str, optional): The name of the model to use for transcription.
            language (Optional[str], optional): The language of the audio input.
            device (str, optional): The device to use for processing, either "cpu" or "gpu".
            temperature (float, optional): The sampling temperature during decoding.
            sample_len (Optional[int], optional): The maximum length of the decoded audio.
            best_of (Optional[int], optional): The number of samples to generate and select the best from.
            beam_size (Optional[int], optional): The size of the beam search during decoding.
            patience (Optional[float], optional): The patience value for early stopping during decoding.
            length_penalty (Optional[float], optional): The length penalty applied during decoding.
            prompt (Optional[Union[str, List[int]]], optional): The prompt for the model during decoding.
            prefix (Optional[Union[str, List[int]]], optional): The prefix for the model during decoding.
            suppress_tokens (Optional[Union[str, Iterable[int]]], optional): The tokens to suppress during decoding.
            suppress_blank (bool, optional): Whether to suppress blank tokens during decoding.
            without_timestamps (bool, optional): Whether to generate timestamps during decoding.
            max_initial_timestamp (Optional[float], optional): The maximum initial timestamp during decoding.
            fp16 (bool, optional): Whether to use half-precision floating-point format to perform inference.

        Raises:
            AssertionError: If the model name is not valid.
        """

        super(WhisperLearner, self).__init__()

        # assert model_name in whisper.available_models(), f"Model name: {model_name} is not valid; available models = {whisper.available_models()}"

        self.task = "transcribe"
        self.model_name = None
        self.language = language
        self.device = device
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature = temperature
        self.sample_len = sample_len
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.prompt = prompt
        self.prefix = prefix
        self.suppress_tokens = suppress_tokens
        self.suppress_blank = suppress_blank
        self.without_timestamps = without_timestamps
        self.max_initial_timestamp = max_initial_timestamp
        self.fp16 = fp16
        self.sample_rate = 16000

        if self.device == "cpu" and self.fp16:
            logger.warning("FP16 is not supported on CPU, using FP32 instead.")
            self.fp16 = False

        self.decode_options = whisper.DecodingOptions(
            task=self.task,
            language=self.language,
            temperature=self.temperature,
            sample_len=self.sample_len,
            best_of=self.best_of,
            beam_size=self.beam_size,
            patience=self.patience,
            length_penalty=self.length_penalty,
            # prompt=self.prompt,
            prefix=self.prefix,
            suppress_tokens=self.suppress_tokens,
            suppress_blank=self.suppress_blank,
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            fp16=self.fp16,
        )


    def save(self, path: Union[str, Path]):
        """
        Save model weights to path

        Args:
            path (Union[str, Path]): Directory in which to save model weights. 
        """
        pass

    def load(
        self,
        name: str,
        download_dir: Union[str, Path] = "./",
        in_memory: bool = False,
    ):
        """
        Adapted from Whisper load_model method: https://github.com/openai/whisper/blob/main/whisper/__init__.py#L97
        """

        self.model_name = name
        self.download_dir = download_dir

        self.model = whisper.load_model(name=self.model_name, device=self.device, download_root=self.download_dir, in_memory=in_memory)


    def download(self,
        name: str,
        download_dir: str = None,
     ):
        """
        Adapted from Whisper load_model method: https://github.com/openai/whisper/blob/main/whisper/__init__.py#L97
        """

        download_root = download_dir
        in_memory = False

        if download_root is None:
            default = os.path.join(os.path.expanduser("~"), ".cache")
            download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

        if name in MODELS_URL:
            whisper._download(MODELS_URL[name], download_root, in_memory)
        elif os.path.isfile(name):
            raise RuntimeError(
                f"Model {name} should not be a path."
            )
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {whisper.available_models()}"
            )


    def reset(self):
        return

    def fit(self):
        return

    def eval(self, dataset: Dataset, batch_size: int = 2):
        """
        Evaluate the model on the given dataset.

        """
        raise NotImplementedError

    def infer(
        self,
        batch: Union[Timeseries, np.ndarray, torch.Tensor, str],
        builtin_transcribe: bool = True,
    ) -> Union[WhisperTranscription, List[WhisperTranscription]]:
        """
        Run inference on a batch of audio sample.

        Args:
            batch (Union[Timeseries, np.ndarray, torch.Tensor]): The audio sample as a Timeseries, torch.Tensor, or np.ndarray.

        Returns:
            List[Dict]: The inference results.

        Raises:
            TypeError: If the input batch is not a Timeseries, torch.Tensor, or np.ndarray.
        """
        if isinstance(batch, Timeseries):
            data = batch.numpy().reshape(-1)
        elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
            data = batch
        elif isinstance(batch, str):
            data = whisper.load_audio(batch)
        else:
            raise TypeError("batch must be a timeseries, torch.tensor or np.ndarray")

        if builtin_transcribe:
            decode_results = self.model.transcribe(data, no_speech_threshold=self.no_speech_threshold, logprob_threshold=self.logprob_threshold, condition_on_previous_text=self.condition_on_previous_text, **asdict(self.decode_options))
            # print(decode_results)
            return WhisperTranscription(text=decode_results["text"], segments=decode_results["segments"])
        else:
            mel = self.preprocess(data)
            decode_results = self.model.decode(mel=mel, options=self.decode_options)
            output = self.postprocess(decode_results)

        return output

    def preprocess(self, data: Union[np.array, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess audio data.
        """

        data = whisper.pad_or_trim(data)
        mel = whisper.log_mel_spectrogram(data, device=self.device)

        return mel

    def postprocess(self, decode_results: Union[DecodingResult, List[DecodingResult]]) -> Union[WhisperTranscription, List[WhisperTranscription]]:
        """
        Postprocess the decoding results.
        """       

        # Ensure we always work with a list
        if not isinstance(decode_results, list):
            decode_results = [decode_results]

        results = [
            WhisperTranscription(text=result["text"], segments=result["segments"])
            for result in decode_results
        ]
        return results[0] if len(results) == 1 else results

    @staticmethod
    def load_audio(file: str) -> np.ndarray:
        return whisper.load_audio(file)

    def optimize(self):
        return

    def _save_onnx(self):
        return
