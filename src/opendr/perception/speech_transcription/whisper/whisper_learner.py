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

import os
from dataclasses import asdict
from logging import getLogger
from typing import Iterable, List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import jiwer

import torch
from torch.utils.data import DataLoader

import whisper
from whisper import _MODELS as MODELS_URL

from opendr.engine.data import Timeseries
from opendr.engine.datasets import DatasetIterator
from opendr.engine.learners import Learner
from opendr.engine.target import WhisperTranscription


logger = getLogger(__name__)


class WhisperLearner(Learner):
    def __init__(
        self,
        verbose: Optional[bool]=None,
        temperature: Union[float, Tuple[float, ...]]=(0.0, 0.2, 0.4, 0.6, 0.8, 1),
        compression_ratio_threshold: Optional[float]=2.4,
        logprob_threshold: Optional[float]=-0.8,
        no_speech_threshold: Optional[float]=0.6,
        condition_on_previous_text: bool=False,
        word_timestamps: bool=False,
        prepend_punctuations: str="\"'“¿([{-",
        append_punctuations: str="\"'.。,，!！?？:：”)]}、",
        language: Optional[str]="en",
        sample_len: Optional[int]=None,
        best_of: Optional[int]=None,
        beam_size: Optional[int]=None,
        patience: Optional[float]=None,
        length_penalty: Optional[float]=None,
        prompt: Optional[Union[str, List[int]]]=None,
        prefix: Optional[Union[str, List[int]]]=None,
        suppress_tokens: Optional[Union[str, Iterable[int]]]="-1",
        suppress_blank: bool=True,
        without_timestamps: bool=False,
        max_initial_timestamp: Optional[float]=1.0,
        fp16: bool=True,
        device: str="cuda",
    ):
        """
        Initialize transcription model that uses Whisper.

        Args:
            Transribe parameters: The following parameters is use in the built-in transcribe function of Whisper.

            verbose: bool
                Whether to display the text being decoded to the console. If True, displays all the details,
                If False, displays minimal details. If None, does not display anything.

            temperature: Union[float, Tuple[float, ...]]
                Temperature for sampling. It can be a tuple of temperatures, which will be successively used
                upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

            compression_ratio_threshold: Optional[float]
                If the gzip compression ratio is above this value, treat as failed.

            logprob_threshold: Optional[float]
                If the average log probability over sampled tokens is below this value, treat as failed

            no_speech_threshold: Optional[float]
                If the no_speech probability is higher than this value AND the average log probability over sampled tokens is
                below `logprob_threshold`, consider the segment as silent

            condition_on_previous_text: bool
                If True, the previous output of the model is provided as a prompt for the next window;
                disabling may make the text inconsistent across windows, but the model becomes less prone to
                getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

            word_timestamps: bool
                Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
                and include the timestamps for each word in each segment.

            prepend_punctuations: str
                If word_timestamps is True, merge these punctuation symbols with the next word.

            append_punctuations: str
                If word_timestamps is True, merge these punctuation symbols with the previous word.

            Decode parameters: The following parameters is use in the decode process.

            language: Optional[str]
                Language spoken in the audio, specify None to perform language detection.

            sample_len: Optinal[int]
                Maximum number of tokens to sample.

            best_of: Optional[int]
                Number of candidates when sampling with non-zero temperature.

            beam_size: Optional[int]
                Number of beams in beam search, only applicable when temperature is zero.

            patience: Optional[float]
                Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is
                equivalent to conventional beam search.

            length_penalty: Optional[float]
                Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length
                normalization by default.

            prompt: Optional[Union[str, List[int]]]
                Text or tokens to feed as the prompt; for more info:
                https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

            prefix: Optional[Union[str, List[int]]]
                Text or tokens to feed as the prefix; for more info:
                https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

            suppress_tokens: Optional[Union[str, Iterable[int]]]
                Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except
                common punctuations.

            suppress_blank: bool
                Suppress blank outputs.

            without_timestamps: bool
                Use <|notimestamps|> to sample text tokens only, the timestamp will be multiple of 30 seconds if the audio file
                is longer than 30 seconds.

            max_initial_timestamp: Optional[float]
                Limit the range of timestamp tokens that can be generated at the beginning of a sequence.

            fp16: bool
                Whether to perform inference in fp16. fp16 is not available on CPU.

            Other parameters:

            device: str
                Device to use for PyTorch inference, either "cpu" or "cuda".
        """

        super(WhisperLearner, self).__init__()

        # Parameters for transcribe function of Whisper.
        self.verbose = verbose
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.word_timestamps = word_timestamps
        self.prepend_punctuations = prepend_punctuations
        self.append_punctuations = append_punctuations

        # Parameters for decoding process.
        self.task = "transcribe"
        self.language = language
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

        # Other parameters.
        self.model_name = None
        self.sample_rate = 16000
        self.device = device

        if self.device == "cpu" and self.fp16:
            logger.warning("FP16 is not supported on CPU, using FP32 instead.")
            self.fp16 = False

        self.decode_options = whisper.DecodingOptions(
            task=self.task,
            language=self.language,
            temperature=self.temperature if isinstance(self.temperature, float) else 0.0,
            sample_len=self.sample_len,
            best_of=self.best_of,
            beam_size=self.beam_size,
            patience=self.patience,
            length_penalty=self.length_penalty,
            prompt=self.prompt,
            prefix=self.prefix,
            suppress_tokens=self.suppress_tokens,
            suppress_blank=self.suppress_blank,
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            fp16=self.fp16,
        )

    def load(
        self,
        name: Optional[str] = None,
        model_path: Optional[str] = None,
        download_dir: Optional[str] = None,
        in_memory: bool = False,
    ):
        """
        Loads Whisper model using Whisper builtin load() method. This method will download model if necessary.
        Adapted from Whisper load_model method: https://github.com/openai/whisper/blob/main/whisper/__init__.py#L97

        Args:
            name (str): name of Whisper model. Could be: tiny.en, tiny, base, base.en, etc. Defaults to None.
            model_path (str, optional): path to model checkpoint. Defaults to None.
            download_dir (str, optional): directory to save the downloaded model. Defaults to "./".
            in_memory (bool, optional): whether to load the model in memory. Defaults to False.
        """

        if model_path is not None:
            if os.path.isfile(model_path):
                self.model_name = os.path.splitext(os.path.basename(model_path))[1]
                whisper_path = model_path
            else:
                raise ValueError(f"{name} is not a correct path to a model checkpoint.")
        elif name is not None:
            if name in whisper.available_models():
                self.model_name = name
                whisper_path = name
            else:
                raise ValueError(f"{name} is not a correct Whisper model name.")
        else:
            raise ValueError("Please specify a model name or path to model checkpoint.")

        self.model = whisper.load_model(
            name=whisper_path,
            device=self.device,
            download_root=download_dir,
            in_memory=in_memory,
        )

    def download(
        self,
        name: str,
        download_dir: Optional[str] = None,
    ):
        """
        Download Whisper model.
        Adapted from Whisper load_model method: https://github.com/openai/whisper/blob/main/whisper/__init__.py#L97

        Args:
            name (str): name of model.
            download_dir (str, optional): directory to save the downloaded model. Defaults to "./".
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

    def infer(
        self,
        audio: Union[Timeseries, np.ndarray, torch.Tensor, str],
        initial_prompt: Optional[str] = None,
    ) -> WhisperTranscription:
        """
        Run inference on an audio sample. Please call the load() method before calling this method.

        Args:
            audio (Union[Timeseries, np.ndarray, torch.Tensor, str]): The audio sample as a Timeseries, torch.Tensor, or
            np.ndarray or a string of file path.

            initial_prompt (str, optional):  Optional text to provide as a prompt for the first window.
            This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or
            proper nouns to make it more likely to predict those word correctly.

        Returns:
            WhisperTranscription: Transcription results with side information.

        Raises:
            TypeError: If the input batch is not a Timeseries, torch.Tensor, np.ndarray or str.
        """

        if isinstance(audio, Timeseries):
            data = audio.numpy().reshape(-1)
        elif isinstance(audio, (torch.Tensor, np.ndarray)):
            data = audio
        elif isinstance(audio, str):
            data = whisper.load_audio(audio)
        else:
            raise TypeError("batch must be a timeseries, torch.tensor or np.ndarray")

        decode_results = self.model.transcribe(
            data,
            verbose=self.verbose,
            compression_ratio_threshold=self.compression_ratio_threshold,
            no_speech_threshold=self.no_speech_threshold,
            logprob_threshold=self.logprob_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=self.word_timestamps,
            initial_prompt=initial_prompt,
            prepend_punctuations=self.prepend_punctuations,
            append_punctuations=self.append_punctuations,
            **asdict(self.decode_options),
        )
        return WhisperTranscription(
            text=decode_results["text"], segments=decode_results["segments"]
        )

    @staticmethod
    def load_audio(file: str) -> np.ndarray:
        """
        Load audio from a file.

        Args
            file (str): Path to audio file.

        Returns:
            np.ndarray: Audio data.
        """
        return whisper.load_audio(file)

    def save(self):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def reset(self):
        """
        Set Whisper model and model name attributes to None. Use before loading a new model.
        """
        self.model_name = None
        self.model = None

    def fit(self):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def eval(self, dataset: DatasetIterator, save_path_csv: Optional[str] = None) -> Dict:
        """
        Evaluate Whisper model on the given dataset.

        Args:
            dataset (DatasetIterator): A speech dataset.
            save_path_csv (str, optional): The path to save the evaluation results.

        Returns:
            Dict: A dictionary containing the word error rate (WER).

        Raises:
            AssertionError: If the model is not loaded.
        """

        assert self.model is not None, "Model is not loaded. Please load a model before evaluating."

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        hypotheses = []
        references = []

        with torch.no_grad():
            for audio, text in tqdm(loader):
                # Remove batch dimension
                audio = audio[0]
                text = text[0]

                result = self.infer(audio)

                hypotheses.append(result.text.lower())
                references.append(text.lower())

            transcriptions = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
            if save_path_csv is not None:
                transcriptions.to_csv(save_path_csv, index=False)

        wer = jiwer.wer(list(transcriptions["reference"]), list(transcriptions["hypothesis"]))

        return {"wer": wer}

    def optimize(self):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def _save_onnx(self):
        """This method is not used in this implementation."""
        raise NotImplementedError
