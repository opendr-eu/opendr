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

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import whisper
from whisper import _MODELS as MODELS_URL
from whisper.model import ModelDimensions, Whisper
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

from opendr.engine.data import Timeseries
from opendr.engine.target import Transcription
from opendr.engine.learners import Learner
from opendr.perception.speech_recognition.whisper.algorithm.utils import matching_percentage


logger = getLogger(__name__)


class WhisperLearner(Learner):
    def __init__(
        self,
        model_name: str,
        language: Optional[str] = "en",
        keywords_list: Optional[List[str]] = None,
        device: str = "cpu",
        temperature: float = 0.0,
        sample_len: Optional[int] = None,
        best_of: Optional[int] = None,
        beam_size: Optional[int] = None,
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
            keywords_list (Optional[List[str]], optional): A list of keywords registered by user.
            normalized_text (Optional[bool], optional): Whether to normalize the transcribed text.
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

        assert model_name in whisper.available_models(), f"Model name: {model_name} is not valid; available models = {whisper.available_models()}"

        self.task = "transcribe"
        self.model_name = model_name
        self.language = language
        self.keywords_list = keywords_list
        self.device = device
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
            prompt=self.prompt,
            prefix=self.prefix,
            suppress_tokens=self.suppress_tokens,
            suppress_blank=self.suppress_blank,
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            fp16=self.fp16,
        )

        self.basic_text_normalizer = BasicTextNormalizer()


    def _load_model_weights(
        self,
        load_path: Union[str, Path],
        in_memory: bool = False,
    ) -> torch.nn.Module:
        """
        Load pretrained model weights.

        Args:
            load_path (Union[str, Path]): The path to the pretrained model weights.
            in_memory (bool, optional): Whether to load the model weights into memory.

        Returns:
            torch.nn.Module: The loaded model with pretrained weights.

        Raises:
            RuntimeError: If the specified model is not found.
        """

        alignment_heads = whisper._ALIGNMENT_HEADS[self.model_name]

        if os.path.isfile(load_path):
            checkpoint_file = open(load_path, "rb").read() if in_memory else load_path
        else:
            raise RuntimeError(
                f"Model {load_path} not found; available models = {whisper.available_models()}"
            )

        with (
            io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
        ) as fp:
            checkpoint = torch.load(fp, map_location=self.device)
        del checkpoint_file

        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_alignment_heads(alignment_heads)

        return model.to(self.device)

    def save(self, path: Union[str, Path]):
        """
        Save model weights to path

        Args:
            path (Union[str, Path]): Directory in which to save model weights. 
        """
        pass

    def load(
        self,
        load_path: Optional[Union[str, Path]]= None,
        download_root: Union[str, Path] = "./",
        in_memory: bool = False,
    ):
        """
        Load a pretrained model from the specified path or download it if not provided.

        Args:
            load_path (Union[str, Path], optional): The path to the pretrained model weights.
            download_root (Union[str, Path], optional): The root directory for downloading the pretrained model.
            in_memory (bool, optional): Whether to load the model weights into memory.

        Returns:
            None

        Raises:
            AssertionError: If download_root is not specified when load_path is None.
            AssertionError: If the model name from the given path does not match the current model name.
        """

        if load_path is None:
            assert download_root is not None, "download_root must be specified when load_path is None"
            self.download(path=download_root)

            url = MODELS_URL[self.model_name]
            load_path = os.path.join(download_root, os.path.basename(url))
        else:
            load_model_name, _ = os.path.splitext(os.path.basename(load_path))
            assert self.model_name == load_model_name, f"Given path: {load_path} has model name does not match with the model name: {self.model_name}"

        self.model = self._load_model_weights(
            load_path=load_path,
            in_memory=in_memory,
        )

        self.tokenizer = get_tokenizer(self.model.is_multilingual, language=self.language, task=self.task)

    def download(self, path: Union[str, Path] = "."):
        """
        Download the pretrained model specified by the current model name.

        Args:
            path (Union[str, Path], optional): The path to save the downloaded pretrained model.

        Raises:
            RuntimeError: If the target path for the downloaded model is not a regular file.
            RuntimeError: If the SHA256 checksum does not match after downloading the model.
        """

        url = MODELS_URL[self.model_name]
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

        with urllib.request.urlopen(url) as source, open(
            download_target, "wb"
        ) as output:
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

    def eval(self, dataset: Dataset, batch_size: int = 2, save_path: str = None) -> Dict:
        """
        Evaluate the model on the given dataset.

        Args:
            dataset (Dataset): A speech command dataset.
            batch_size (int, optional): The batch size for DataLoader.
            save_path (str, optional): The path to save the evaluation results.

        Returns:
            Dict: A dictionary containing the evaluation performance metrics.

        Raises:
            AssertionError: If the model is not loaded.
        """
        assert self.model is not None, "Model is not loaded. Please load a model before evaluating."


        def matching_percentage(hypothesis: List[str], reference: List[str]) -> float:
            """
            Compute the accuracy of string predicted by the model and the ground truth.
            Used in keyword matching.

            Args:
                hypothesis (List[str]): A list of predicted strings.
                reference (List[str]): A list of ground truth strings.

            Returns:
                float: The accuracy of the predicted strings.

            Raises:
                AssertionError: If the model is not loaded.
            """

            if len(hypothesis) != len(reference):
                raise ValueError("Both lists must have the same length.")

            matching_count = sum(h == r for h, r in zip(hypothesis, reference))
            total_count = len(hypothesis)

            return matching_count / total_count

        normalizer = EnglishTextNormalizer()
        logger.warning("Used English text normalizer of Whisper to standardize the prediction and reference text.")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model.eval() 
        self.model.to(self.device)

        hypotheses = []
        references = []

        with torch.no_grad():
            for samples, texts in tqdm(loader):
                results = self.infer(samples)

                if not isinstance(results, list):
                    results = [results]

                texts = [normalizer(text) for text in texts]
                prediction = [normalizer(result.data) for result in results]

                hypotheses.extend(prediction)
                references.extend(texts)

            data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

            if save_path is not None:
                data.to_csv(save_path, index=False)

        mp = matching_percentage(hypothesis=list(data["hypothesis"]), 
                                 reference=list(data["reference"]))
        performance = {"total_accuracy": mp}

        return performance


    def infer(
        self,
        batch: Union[Timeseries, np.ndarray, torch.Tensor]
    ) -> List[Dict]:
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
        else:
            raise TypeError("batch must be a Timeseries, torch.Tensor or np.ndarray")

        mel = self.preprocess(data)

        decode_results = self.model.decode(mel=mel, options=self.decode_options)
        output = self.postprocess(decode_results)

        return output

    def preprocess(self, data: Union[np.array, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess audio data.

        This function pads or trims the input audio data and computes the log Mel spectrogram.
        The audio data should be a 1-D array for a single audio or a 2-D array with the first
        dimension being the batch size.

        Args:
            data (Union[np.array, torch.Tensor]): Input audio data.

        Returns:
            torch.Tensor: Log Mel spectrogram of the preprocessed audio data.
        """

        data = whisper.pad_or_trim(data)
        mel = whisper.log_mel_spectrogram(data, device=self.device)

        return mel

    def postprocess(
        self,
        decode_results: Union[whisper.DecodingResult, List[whisper.DecodingResult]],
    ):
        """
        Postprocess the decoding results.

        This function processes the given decoding results, converting them into a list of dictionaries.
        Each dictionary contains information such as text, tokens, temperature, avg_logprob, compression_ratio,
        no_speech_prob, language, and language_probs.

        Args:
            decode_results (Union[whisper.DecodingResult, List[whisper.DecodingResult]]): Decoding results to be postprocessed.

        Returns:
            List[dict]: A list of dictionaries containing postprocessed decoding result information.
        """       
        decode_results = (
            decode_results if isinstance(decode_results, list) else [decode_results]
        )

        results = [
            Transcription(text=self.basic_text_normalizer(self.tokenizer.decode(result.tokens)))
            for result in decode_results
        ]

        return results


    def optimize(self):
        return

    def _save_onnx(self):
        return
