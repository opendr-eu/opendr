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

import unittest
import torch
# import shutil
# Temporary for debugging. TODO: Remove
# import sys
# if "/Users/au478108/Projects/opendr_internal/src" not in sys.path:
#     sys.path.append("/Users/au478108/Projects/opendr_internal/src")

from perception.activity_recognition.x3d.x3d_learner import X3DLearner
from perception.activity_recognition.datasets.kinetics import KineticsDataset

from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)

_DATASET_PATH = Path("/Users/au478108/Projects/datasets/kinetics400micro")
_BACKBONE = "xs"


class TestX3DLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(
            "./tests/sources/tools/perception/activity_recognition/x3d/temp"
        )

        # Download all required files for testing
        X3DLearner.download(path=Path(cls.temp_dir) / "weights", model_weights={_BACKBONE})
        cls.learner = X3DLearner(
            device="cpu", temp_path=str(cls.temp_dir), iters=1, batch_size=2, backbone=_BACKBONE, num_workers=0,
        )

    # @classmethod
    # def tearDownClass(cls):
    #     try:
    #         shutil.rmtree(str(cls.temp_dir))
    #     except OSError as e:
    #         logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_downloaded(self):
        assert Path(self.temp_dir) / "weights" / "x3d_s.pyth"

    def xtest_save_and_load(self):
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

    def xtest_fit(self):
        train_ds = KineticsDataset(path=_DATASET_PATH, frames_per_clip=4, split="train")
        val_ds = KineticsDataset(path=_DATASET_PATH, frames_per_clip=4, split="val")

        # Initialize with random parameters
        self.learner.model = None
        self.learner.init_model()

        # Store prior parameters
        m = list(self.learner.model.parameters())[0].clone()

        # Fit model
        self.learner.fit(dataset=train_ds, val_dataset=val_ds, steps=1)

        # Check that parameters changed
        assert not torch.equal(m, list(self.learner.model.parameters())[0])

    def xtest_eval(self):
        test_ds = KineticsDataset(path=_DATASET_PATH, frames_per_clip=4, split="test")

        self.learner.load_model_weights(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        results = self.learner.eval(test_ds, steps=2)

        assert results["accuracy"] > 0.2  # Most likely ≈ 60%
        assert results["results"] < 20  # Most likely ≈ 6.0

    def test_infer(self):
        ds = KineticsDataset(path=_DATASET_PATH, frames_per_clip=4, split="test")
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))[0]

        self.learner.load_model_weights(self.temp_dir / "weights" / f"x3d_{_BACKBONE}.pyth")
        results = self.learner.infer(batch)

        # Results is a batch with each item summing to 1.0
        assert torch.all(torch.sum(results, dim=1) == torch.ones((2, 1)))

    # def test_save_load_onnx(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.init_model()
    #     self.pose_estimator.optimize()
    #     self.pose_estimator.save(os.path.join(self.temp_dir, "testModel"))
    #     self.pose_estimator.model = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "testModel"))
    #     self.assertIsNotNone(
    #         self.pose_estimator.ort_session,
    #         "ort_session is None after loading onnx model.",
    #     )
    #     # Cleanup
    #     rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))
    #     rmdir(os.path.join(self.temp_dir, "testModel"))

    # def test_optimize(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "trainedModel"))
    #     self.pose_estimator.optimize()
    #     self.assertIsNotNone(self.pose_estimator.ort_session)
    #     # Cleanup
    #     rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))


if __name__ == "__main__":
    unittest.main()
