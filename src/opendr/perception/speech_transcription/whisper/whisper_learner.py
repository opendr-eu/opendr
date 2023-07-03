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


from typing import Union, Iterable, Optional, List
from dataclasses import asdict
from logging import getLogger
import os

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import whisper
from whisper import DecodingResult
from whisper import _MODELS as MODELS_URL

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
        word_timestamps: bool = False,
        temperature: float = 0.0,
        compression_ratio_threshold: Optional[float] = 2.4,
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

            temperature: Union[float, Tuple[float, ...]]
                Temperature for sampling. It can be a tuple of temperatures, which will be successively used
                upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
                When use infer method with builtin_transcribe=False, the temperature must be a number.

            compression_ratio_threshold: float
                If the gzip compression ratio is above this value, treat as failed.

            logprob_threshold (float):
                If the average log probability over sampled tokens is below this value, treat as failed

            no_speech_threshold (float):
                If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `logprob_threshold`, consider the segment as silent

            condition_on_previous_text: bool
                If True, the previous output of the model is provided as a prompt for the next window;
                disabling may make the text inconsistent across windows, but the model becomes less prone to
                getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

            word_timestamps: bool
                Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
                and include the timestamps for each word in each segment.





            device (str):
                device to use for PyTorch inference, either "cpu" or "gpu".



            Decode parameters:

            language (Optional[str]):
                language spoken in the audio, specify None to perform language detection.

            temperature (Optional[float]):
                When using infer method with builtin_transcribe=True, we call the transcribe function of Whisper, which
                iteratively set the value of temperature from the given tuple or float.

            sample_len (Optinal[int]):
                Maximum number of tokens to sample.

            best_of (Optional[int]):
                Number of candidates when sampling with non-zero temperature.

            beam_size (Optional[int]):
                Number of beams in beam search, only applicable when temperature is zero.

            patience (Optional[float]):
                Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search.

            length_penalty (Optional[float]):
                Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default.

            prompt (Optional[Union[str, List[int]]]):
                Text or tokens to feed as the prompt; for more info: # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

            prefix (Optional[Union[str, List[int]]]):
                Text or tokens to feed as the prefix; for more info: # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

            suppress_tokens (Optional[Union[str, Iterable[int]]]):
                Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations.

            suppress_blank (bool):
                Suppress blank outputs.

            without_timestamps (bool):
                Use <|notimestamps|> to sample text tokens only, the timestamp will be multiple of 30 seconds if the audio file is longer than 30 seconds.

            max_initial_timestamp (Optional[float])

            fp16 (bool):
                whether to perform inference in fp16. fp16 is not available on CPU.

        Raises:
            AssertionError: If the model name is not valid.
        """

        super(WhisperLearner, self).__init__()

        self.task = "transcribe"
        self.model_name = None
        self.language = language
        self.device = device
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.word_timestamps = word_timestamps
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
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

    def save(self):
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

        if name is None:
            raise ValueError("Please specify a model name or path to model checkpoint.")

        self.model_name = name
        self.download_dir = download_dir

        self.model = whisper.load_model(
            name=self.model_name,
            device=self.device,
            download_root=self.download_dir,
            in_memory=in_memory,
        )

    def download(
        self,
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
            download_root = os.path.join(
                os.getenv("XDG_CACHE_HOME", default), "whisper"
            )

        if name in MODELS_URL:
            whisper._download(MODELS_URL[name], download_root, in_memory)
        elif os.path.isfile(name):
            raise RuntimeError(f"Model {name} should not be a path.")
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
            decode_results = self.model.transcribe(
                data,
                compression_ratio_threshold=self.compression_ratio_threshold,
                no_speech_threshold=self.no_speech_threshold,
                logprob_threshold=self.logprob_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
                word_timestamps=self.word_timestamps,
                **asdict(self.decode_options),
            )
            print(decode_results)
            return WhisperTranscription(
                text=decode_results["text"], segments=decode_results["segments"]
            )
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

    def postprocess(
        self, decode_results: Union[DecodingResult, List[DecodingResult]]
    ) -> Union[WhisperTranscription, List[WhisperTranscription]]:
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
