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
import time
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



class SubsetSC(SPEECHCOMMANDS):
    def __init__(
        self,
        root: str = "./",
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        subset: str = "testing",
        device: Optional[Union[str, torch.device]] = "cpu",
    ):

        super().__init__(
            root=root, url=url, folder_in_archive=folder_in_archive, subset=subset
        )
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

        sample = whisper.pad_or_trim(audio.flatten().to(self.device))

        return (sample, label)


def main(args):

    test_set = SubsetSC(
        root=args.root,
        url=args.url,
        folder_in_archive=args.folder_in_archive,
        subset=args.subset,
        device=args.device,
    )

    keywords_list = [
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

    learner = WhisperLearner(model_name=args.model_name, keywords_list=keywords_list, normalized_text=True, fp16=args.fp16, device=args.device)

    learner.load(args.load_path)  # Auto download the model to currenct directory and load it.
    
    performance = learner.eval(dataset=test_set, batch_size=args.batch_size, save_path=f"./{args.model_name}_fp16_{args.fp16}.csv")

    # Print results.
    for word, mp in performance["word_accuracy"].items(): 
        print(f"Maching percentage for '{word}': {mp * 100:.2f} % ")

    print(f"Maching percentage for all keywords: {performance['total_accuracy'] * 100:.2f} %")


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

