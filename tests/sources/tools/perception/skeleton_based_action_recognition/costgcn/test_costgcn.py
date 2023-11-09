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
import unittest
import shutil

from opendr.perception.skeleton_based_action_recognition import CoSTGCNLearner
from opendr.engine.datasets import ExternalDataset
# from opendr.engine.target import Category
from pathlib import Path
from logging import getLogger

device = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

logger = getLogger(__name__)

_BACKBONE = "costgcn"


class TestCoSTGCNLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST Continual STGCN Learner\n"
            "**********************************"
        )
        cls.temp_dir = Path("./tests/sources/tools/perception/skeleton_based_action_recognition/temp")

        cls.learner = CoSTGCNLearner(
            device=device,
            temp_path=str(cls.temp_dir),
            iters=1,
            batch_size=2,
            backbone=_BACKBONE,
            num_workers=0,
        )

        # Download all required files for testing
        cls.pretrained_weights_path = cls.learner.download(
            path=os.path.join(cls.temp_dir, "pretrained_models"),
            method_name="costgcn",
            mode="pretrained",
            file_name="costgcn_ntu60_xview_joint.ckpt",
        )
        cls.Train_DATASET_PATH = cls.learner.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data")
        )
        cls.Val_DATASET_PATH = cls.learner.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data")
        )

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(str(cls.temp_dir))
        except OSError as e:
            logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_fit(self):
        print(
            "\n\n**********************************\nTest CoSTGCNLearner fit \n*"
            "*********************************"
        )

        train_ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="train_joints.npy",
            labels_filename="train_labels.pkl",
            skeleton_data_type="joint",
            phase="train",
            verbose=False,
        )

        val_ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )

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
        test_ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )

        self.learner.load(self.pretrained_weights_path)
        results = self.learner.eval(test_ds, steps=2)

        assert results["accuracy"] > 0.5
        assert results["loss"] < 1

    def test_infer(self):
        ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.learner.batch_size, num_workers=0)
        batch = next(iter(dl))[0]
        frame = batch[:, :, -1]  # Select a single frame

        self.learner.model.clean_state()
        self.learner.model.forward_steps(batch[:, :, :-1])  # Init model state

        # Input is Tensor
        results1 = self.learner.infer(frame)
        # Results is a batch with each item summing to 1.0
        assert all([torch.isclose(torch.sum(r.confidence), torch.tensor(1.0)) for r in results1])

    # DISABLED: test passes however hangs unittest, preventing it from completing
    # def test_optimize(self):
    #    self.learner.batch_size = 2
    #    self.learner._ort_session = None
    #    self.learner.optimize()
    #    step_input = self.learner._example_input[:, :, 0]
    #    step_output = self.learner.infer(step_input)
    #    assert isinstance(step_output[0], Category)
    #
    #    assert self.learner._ort_session is not None
    #
    #    # Clean up
    #    self.learner._ort_session = None

    def test_save_and_load(self):
        assert self.learner.model is not None
        self.learner.batch_size == 2
        self.learner.save(self.temp_dir)
        # Make changes to check subsequent load
        self.learner.model = None
        self.learner.batch_size = 42
        self.learner.load(self.temp_dir)
        self.assertIsNotNone(self.learner.model, "model is None after loading pth model.")
        assert self.learner.batch_size == 2


if __name__ == "__main__":
    unittest.main()
