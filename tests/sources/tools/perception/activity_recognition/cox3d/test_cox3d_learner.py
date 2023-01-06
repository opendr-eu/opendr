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

import shutil
import torch
import unittest
import numpy as np
import os

from opendr.perception.activity_recognition import CoX3DLearner
from opendr.perception.activity_recognition import KineticsDataset
from opendr.engine.data import Image
from opendr.engine.target import Category
from pathlib import Path
from logging import getLogger

device = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

logger = getLogger(__name__)

_BACKBONE = "xs"


class TestCoX3DLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Continual Activity Recognition CoX3D Learner\n"
              "**********************************")
        cls.temp_dir = Path("./tests/sources/tools/perception/activity_recognition/temp")

        # Download model weights
        CoX3DLearner.download(path=Path(cls.temp_dir) / "weights", model_names={_BACKBONE})
        cls.learner = CoX3DLearner(
            device=device, temp_path=str(cls.temp_dir), iters=1, batch_size=2, backbone=_BACKBONE, num_workers=0,
        )

        # Download mini dataset
        cls.dataset_path = cls.temp_dir / "datasets" / "kinetics3"
        KineticsDataset.download_micro(cls.temp_dir / "datasets")
        cls.train_ds = KineticsDataset(path=cls.dataset_path, frames_per_clip=4, split="train", spatial_pixels=160)
        cls.val_ds = KineticsDataset(path=cls.dataset_path, frames_per_clip=4, split="val", spatial_pixels=160)
        cls.test_ds = KineticsDataset(path=cls.dataset_path, frames_per_clip=4, split="test", spatial_pixels=160)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(str(cls.temp_dir))
        except OSError as e:
            logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_downloaded(self):
        assert Path(self.temp_dir) / "weights" / "x3d_xs.pyth"

    def test_save_and_load(self):
        assert self.learner.model is not None
        self.learner.save(self.temp_dir)
        # Make changes to check subsequent load
        self.learner.model = None
        self.learner.batch_size = 42
        self.learner.load(self.temp_dir)
        self.assertIsNotNone(
            self.learner.model, "model is None after loading pth model."
        )
        assert self.learner.batch_size == 2

    def test_fit(self):
        # Initialize with random parameters
        self.learner.model = None
        self.learner.init_model()

        # Store prior parameters
        m = list(self.learner.model.parameters())[0].clone()

        # Fit model
        self.learner.fit(dataset=self.train_ds, val_dataset=self.val_ds, steps=1)

        # Check that parameters changed
        assert not torch.equal(m, list(self.learner.model.parameters())[0])

    def test_eval(self):
        self.learner.model.clean_state()
        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        results = self.learner.eval(self.test_ds, steps=2)

        assert results["accuracy"] > 0.5
        assert results["loss"] < 20

    def test_infer(self):
        dl = torch.utils.data.DataLoader(self.test_ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))[0]
        batch = batch[:, :, 0]  # Select a single frame

        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")

        # Warm up
        self.learner.model.forward_steps(
            batch.unsqueeze(2).repeat(1, 1, self.learner.model.receptive_field - 1, 1, 1)
        )

        # Input is Tensor
        results1 = self.learner.infer(batch)
        # Results is a batch with each item summing to 1.0
        assert all([torch.isclose(torch.sum(r.confidence), torch.tensor(1.0)) for r in results1])

        # Input is Image
        results2 = self.learner.infer([Image(batch[0], dtype=np.float64), Image(batch[1], dtype=np.float32)])
        assert results1[0].data == results2[0].data
        assert results1[1].data == results2[1].data

        # Input is List[Image]
        results3 = self.learner.infer([Image(v, dtype=np.float64) for v in batch])
        assert results1[0].data == results3[0].data
        assert results1[1].data == results3[1].data

    def test_optimize(self):
        self.learner.ort_session = None
        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        self.learner.optimize()

        assert self.learner._ort_session is not None

        step_input = self.learner._example_input.repeat(
            self.learner.batch_size, 1, 1, 1
        )
        step_output = self.learner.infer(step_input)
        assert isinstance(step_output[0], Category)

        # Clean up
        self.learner.ort_session = None


if __name__ == "__main__":
    unittest.main()
