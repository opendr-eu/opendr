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

import continual
import os
import shutil
import torch
import unittest

from opendr.perception.activity_recognition import CoTransEncLearner
from opendr.engine.data import Vector, Timeseries
from opendr.engine.target import Category
from opendr.perception.activity_recognition.datasets import DummyTimeseriesDataset
from pathlib import Path
from logging import getLogger
import onnxruntime as ort

device = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

logger = getLogger(__name__)

_BATCH_SIZE = 1


class TestCoTransEncLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST Continual Transformer Encoder Learner\n"
            "**********************************"
        )
        cls.temp_dir = Path("./tests/sources/tools/perception/activity_recognition/temp")

        cls.learner = CoTransEncLearner(
            batch_size=_BATCH_SIZE,
            device=device,
            input_dims=8,
            hidden_dims=32,
            sequence_len=64,
            num_heads=8,
            num_classes=4,
            temp_path=str(cls.temp_dir),
        )

        cls.train_ds = DummyTimeseriesDataset(sequence_len=64, num_sines=8, num_datapoints=128)
        cls.val_ds = DummyTimeseriesDataset(
            sequence_len=64, num_sines=8, num_datapoints=128, base_offset=64
        )

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(str(cls.temp_dir))
        except OSError as e:
            logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_save_and_load(self):
        assert self.learner.model is not None
        self.learner.save(self.temp_dir)
        # Make changes to check subsequent load
        self.learner.model = None
        self.learner.batch_size = 42
        self.learner.load(self.temp_dir)
        self.assertIsNotNone(self.learner.model, "model is None after loading pth model.")
        assert self.learner.batch_size == _BATCH_SIZE

    def test_fit(self):
        # Initialize with random parameters
        self.learner.model = None
        self.learner.init_model()

        # Store prior parameters
        m = list(self.learner.model.parameters())[0].clone()

        # Fit model
        self.learner.fit(dataset=self.train_ds, val_dataset=self.val_ds, steps=2)

        # Check that parameters changed
        assert not torch.equal(m, list(self.learner.model.parameters())[0])

    def test_eval(self):
        results = self.learner.eval(self.val_ds, steps=2)

        assert isinstance(results["accuracy"], float)
        assert isinstance(results["loss"], float)

    def test_infer(self):
        dl = torch.utils.data.DataLoader(self.val_ds, batch_size=_BATCH_SIZE, num_workers=0)
        tensor = next(iter(dl))[0][0]

        # Input is Tensor
        results1 = self.learner.infer(tensor.to(device))
        # print(results1)
        # Results has confidence summing to 1.0
        assert torch.isclose(torch.sum(results1.confidence), torch.tensor(1.0))

        # Input is Timeseries
        results2 = self.learner.infer(Timeseries(tensor.permute(1, 0)))
        # print(results2)
        assert torch.allclose(results1.confidence, results2.confidence, atol=1e-2)

        # Input is Vector
        for i in range(64):  # = sequence_len
            results3 = self.learner.infer(Vector(tensor[:, i]))
        assert torch.allclose(results1.confidence, results3.confidence, atol=1e-4)

    def test_optimize(self):
        torch_ok = int(torch.__version__.split(".")[1]) >= 10
        co_ok = int(getattr(continual, "__version__", "0.0.0").split(".")[0]) >= 1
        ort_ok = int(getattr(ort, "__version__", "0.0.0").split(".")[1]) >= 11
        if not (torch_ok and co_ok and ort_ok):
            return  # Skip test

        self.learner._ort_session = None
        self.learner.optimize()
        step_input = self.learner._example_input[:, :, 0]
        step_output = self.learner.infer(step_input)
        assert isinstance(step_output, Category)

        assert self.learner._ort_session is not None
        self.learner._ort_session = None  # Clean up


def rmfile(path):
    try:
        os.remove(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == "__main__":
    unittest.main()
