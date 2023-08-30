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
import tempfile
import unittest

import torch

import numpy as np
from opendr.engine.data import Timeseries
from opendr.engine.datasets import DatasetIterator

from opendr.perception.speech_transcription import VoskLearner
from opendr.engine.target import VoskTranscription


device = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

if device == "cuda":
    device = f"cuda:{torch.cuda.current_device()}"


TEST_MODEL_NAME = "vosk-model-small-en-us-0.15"
TEST_MODEL_LANGUAGE = "en-in"
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
            "\n\n*************************************\nTEST Speech transcription VoskLearner\n"
            "*************************************"
        )
        cls.learner = VoskLearner()

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_eval(self):
        temp_dir = tempfile.TemporaryDirectory()

        self.learner.load(name=TEST_MODEL_NAME, download_dir=temp_dir.name)

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
        audio_bytes = (
            np.random.uniform(-32767, 32767, TEST_SIGNAL_LENGTH)
            .astype(np.int16)
            .tobytes()
        )

        audio_dict = {
            "torch": audio_torch,
            "numpy": audio_numpy,
            "timeseries": audio_timeseries,
            "bytes": audio_bytes,
        }

        self.learner.load(name=TEST_MODEL_NAME, download_dir=temp_dir.name)

        for input_type, audio in audio_dict.items():
            transcription = self.learner.infer(audio)
            self.assertTrue(
                isinstance(transcription, VoskTranscription),
                f"Error when processing input audio of type: {input_type}",
            )

        temp_dir.cleanup()

    def test_load(self):
        temp_dir = tempfile.TemporaryDirectory()

        self.learner.load(name=TEST_MODEL_NAME, download_dir=temp_dir.name)
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, TEST_MODEL_NAME)))
        self.assertTrue(self.learner.model is not None)
        self.assertTrue(self.learner.rec is not None)

        self.learner.reset()
        self.assertTrue(self.learner.model_name is None)
        self.assertTrue(self.learner.language is None)
        self.assertTrue(self.learner.model is None)
        self.assertTrue(self.learner.rec is None)

        self.learner.load(model_path=os.path.join(temp_dir.name, TEST_MODEL_NAME))
        self.assertTrue(self.learner.model is not None)
        self.assertTrue(self.learner.rec is not None)

        self.learner.reset()
        self.assertTrue(self.learner.model_name is None)
        self.assertTrue(self.learner.language is None)
        self.assertTrue(self.learner.model is None)
        self.assertTrue(self.learner.rec is None)

        self.learner.load(language=TEST_MODEL_LANGUAGE, download_dir=temp_dir.name)
        self.assertTrue(self.learner.model is not None)
        self.assertTrue(self.learner.rec is not None)

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(warnings="ignore")
