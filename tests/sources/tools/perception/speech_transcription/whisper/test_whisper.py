# Copyright 2020-2024 OpenDR European Project
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
import wave
import tempfile
import unittest

import torch

import numpy as np
from opendr.engine.data import Timeseries
from opendr.engine.datasets import DatasetIterator

from opendr.perception.speech_transcription import WhisperLearner
from opendr.engine.target import WhisperTranscription


device = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

if device == "cuda":
    device = f"cuda:{torch.cuda.current_device()}"


TEST_MODEL_NAME = "tiny.en"
TEST_SIGNAL_LENGTH = 32000


class DummyDataset(DatasetIterator):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 5

    def __getitem__(self, item):
        return np.ones(TEST_SIGNAL_LENGTH).astype(np.float32), "test transcription"


class WhisperTest(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        print(
            "\n\n****************************************\nTEST Speech transcription WhisperLearner\n"
            "****************************************"
        )
        cls.learner = WhisperLearner(language="en", device=device)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_eval(self):
        temp_dir = tempfile.TemporaryDirectory()

        self.learner.load(name=TEST_MODEL_NAME, download_dir=None)

        eval_dataset = DummyDataset()
        results = self.learner.eval(
            dataset=eval_dataset,
            save_path_csv=os.path.join(temp_dir.name, "eval_results.csv"),
        )
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, "eval_results.csv")))
        self.assertTrue(0.0 <= results["wer"], "Word error rate must be larger than 0.")

    def test_infer(self):
        temp_dir = tempfile.TemporaryDirectory()

        audio_numpy = np.ones(TEST_SIGNAL_LENGTH).astype(np.float32)
        audio_torch = torch.from_numpy(audio_numpy)
        audio_timeseries = Timeseries(np.expand_dims(audio_numpy, axis=0))
        audio_path = os.path.join(temp_dir.name, "random_sound.wav")

        # Generate random data
        data = np.random.uniform(-32767, 32767, TEST_SIGNAL_LENGTH).astype(np.int16)

        # Write data to WAV file
        with wave.open(audio_path, "w") as f:
            f.setnchannels(1)  # mono
            f.setsampwidth(2)  # two bytes (16 bit)
            f.setframerate(16000)
            f.writeframes(data.tobytes())

        audio_dict = {
            "torch": audio_torch,
            "numpy": audio_numpy,
            "timeseries": audio_timeseries,
            "str": audio_path,
        }

        self.learner.load(name=TEST_MODEL_NAME, download_dir=None)

        for input_type, audio in audio_dict.items():
            transcription = self.learner.infer(audio)
            self.assertTrue(
                isinstance(transcription, WhisperTranscription),
                f"Error when processing input audio of type: {input_type}",
            )

        temp_dir.cleanup()

    def test_load(self):
        temp_dir = tempfile.TemporaryDirectory()

        self.learner.load(name=TEST_MODEL_NAME, download_dir=temp_dir.name)
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, f"{TEST_MODEL_NAME}.pt")))
        self.assertTrue(self.learner.model is not None)

        self.learner.reset()
        self.assertTrue(self.learner.model is None)

        self.learner.load(model_path=os.path.join(temp_dir.name, f"{TEST_MODEL_NAME}.pt"))
        self.assertTrue(self.learner.model is not None)

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(warnings="ignore")
