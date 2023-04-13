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


import argparse
from typing import Optional, Union, Tuple, List
import os
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import whisper
from whisper.normalizers import EnglishTextNormalizer
import jiwer

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from opendr.perception.speech_recognition import WhisperLearner


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true'):
        return True
    elif v.lower() in ('False', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def matching_percentage(hypothesis: List[str], reference: List[str]) -> float:
    if len(hypothesis) != len(reference):
        raise ValueError("Both lists must have the same length.")

    matching_count = sum(h == r for h, r in zip(hypothesis, reference))
    total_count = len(hypothesis)

    return matching_count / total_count


class SubsetSC(SPEECHCOMMANDS):
    def __init__(
        self,
        root: str = "./",
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        subset: str = "testing",
        preprocess: bool = True,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):

        super().__init__(
            root=root, url=url, folder_in_archive=folder_in_archive, subset=subset
        )
        self.preprocess = preprocess
        self.device = device

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, i: int) -> Tuple[torch.tensor, str]:
        audio, sample_rate, label, _, _ = super().__getitem__(i)
        assert sample_rate == 16000

        if self.preprocess:
            audio = whisper.pad_or_trim(audio.flatten().to(self.device))
            sample = whisper.log_mel_spectrogram(audio)
        else:
            sample = audio

        return (sample, label)

    @staticmethod
    def get_keywords_list():
        return [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]


def main(args):
    assert args.load_path is not None or args.model_name is not None, "Must specify either --load-path or --model-name."

    test_set = SubsetSC(
        root=args.root,
        url=args.url,
        folder_in_archive=args.folder_in_archive,
        subset=args.subset,
        preprocess=args.preprocess,
        device=args.device,
    )
    batch_size = args.batch_size

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_name = args.model_name
    learner = WhisperLearner()
    learner.load(load_path=args.load_path, model_name=model_name, device=args.device)

    print(
        f"Model is {'multilingual' if learner.model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in learner.model.parameters()):,} parameters."
    )

    if args.preprocess:
        # predict without timestamps for short-form transcription
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=args.fp16)
        infer_fn = partial(learner.model.decode, options=options) 
    else:
        infer_fn = partial(learner.infer, device=args.device, fp16=args.fp16)


    hypotheses = []
    references = []

    for samples, texts in tqdm(loader):
        results = infer_fn(samples)
        if not isinstance(results, list):
            hypotheses.extend([results.text])
        else:
            hypotheses.extend([result.text for result in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    data.to_csv(f"data_{model_name}.csv")

    # data = pd.read_csv(f"data_{args.model_name}.csv")
    # data.fillna("", inplace=True)

    keywords = [normalizer(word) for word in SubsetSC.get_keywords_list()]
    for word in keywords:
        data_filtered = data[data["reference_clean"].apply(lambda x: x == word)]

        mp = matching_percentage(hypothesis=list(data_filtered["hypothesis_clean"]), 
                                 reference=list(data_filtered["reference_clean"]))
   
        print(f"Maching percentage for '{word}': {mp * 100:.2f} % ")

    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    mp = matching_percentage(hypothesis=list(data["hypothesis_clean"]), 
                             reference=list(data["reference_clean"]))
    print(f"Maching percentage for all keywords: {mp * 100:.2f} %")
    print(f"WER (Word error rate): {wer * 100:.2f} %")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper learner class evaluation on Speech Commands dataset")

    parser.add_argument(
        "--root",
        type=str,
        default="./",
        help="Root directory of the speech commands dataset",
    )
    parser.add_argument(
        "--url",
        type=str,
        choices=["speech_commands_v0.01", "speech_commands_v0.02"],
        default="speech_commands_v0.02",
        help="URL of the speech commands dataset",
    )
    parser.add_argument(
        "--folder_in_archive",
        type=str,
        default="SpeechCommands",
        help="Folder name inside the archive of the speech commands dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["testing", "validation", "training"],
        default="testing",
        help="Subset of the dataset to use",
    )
    parser.add_argument(
        "--preprocess",
        type=str2bool,
        default=True,
        help="Transform the audio to log mel spectrogram before feeding to model",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use for processing ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=False,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tiny.en",
        required=False,
        help="Whisper model name",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="Inference with FP16",
    )

    args = parser.parse_args()
    main(args)

