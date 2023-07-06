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

import torch
import torchaudio

from opendr.perception.speech_transcription import (
    VoskLearner,
    WhisperLearner,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000

        return (audio[0], text)


dataset = LibriSpeech("test-clean")

# Load Whisper model.
whisper_learner = WhisperLearner(language="en")
whisper_learner.load(name="tiny.en")

# Load Vosk model.
vosk_learner = VoskLearner()
vosk_learner.load(language="en-us")

learner_dict = {
    "whisper": whisper_learner,
    "vosk": vosk_learner,
}

for name, learner in learner_dict.items():
    results = learner.eval(dataset, save_path_csv=f"./librispeech_{name}.csv")
    print(f"{name.upper()} WER: {results['wer'] * 100:.2f} %")
