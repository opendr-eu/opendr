# Copyright 2020-2021 OpenDR Project
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

from opendr.perception.activity_recognition.cox3d.cox3d_learner import CoX3DLearner
from opendr.perception.activity_recognition.datasets.kinetics import KineticsDataset
from opendr.engine.data import Image
from pathlib import Path
from logging import getLogger

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
            device="cpu", temp_path=str(cls.temp_dir), iters=1, batch_size=2, backbone=_BACKBONE, num_workers=0,
        )

        # Download mini dataset
        cls.dataset_path = cls.temp_dir / "datasets" / "kinetics3"
        KineticsDataset.download_micro(cls.temp_dir / "datasets")

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
        train_ds = KineticsDataset(path=self.dataset_path, frames_per_clip=4, split="train")
        val_ds = KineticsDataset(path=self.dataset_path, frames_per_clip=4, split="val")

        # Initialize with random parameters
        self.learner.model = None
        self.learner.init_model()

        # Store prior parameters
        m = list(self.learner.model.parameters())[0].clone()

        # Fit model
        self.learner.fit(dataset=train_ds, val_dataset=val_ds, steps=1)

        # Check that parameters changed
        assert not torch.equal(m, list(self.learner.model.parameters())[0])

    def test_eval(self):
        test_ds = KineticsDataset(path=self.dataset_path, frames_per_clip=40, split="test")

        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        results = self.learner.eval(test_ds, steps=2)

        assert results["accuracy"] > 0.2
        assert results["loss"] < 20

    def test_infer(self):
        ds = KineticsDataset(path=self.dataset_path, frames_per_clip=4, split="test")
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))[0]
        batch = batch[:, :, 0]  # Select a single frame

        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        self.learner.model.clean_model_state()

        # Input is Tensor
        results1 = self.learner.infer(batch)
        # Results is a batch with each item summing to 1.0
        assert all([torch.isclose(torch.sum(r.confidence), torch.tensor(1.0)) for r in results1])

        # Input is Image
        results2 = self.learner.infer(Image(batch[0], dtype=np.float))
        assert torch.allclose(results1[0].confidence, results2[0].confidence, atol=1e-6)

        # Input is List[Image]
        results3 = self.learner.infer([Image(v, dtype=np.float) for v in batch])
        assert all([torch.allclose(r1.confidence, r3.confidence, atol=1e-6) for (r1, r3) in zip(results1, results3)])

    def test_optimize(self):
        self.learner.ort_session = None
        self.learner.load(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        self.learner.optimize()

        assert self.learner.ort_session is not None

        # Clean up
        self.learner.ort_session = None


if __name__ == "__main__":
    unittest.main()
