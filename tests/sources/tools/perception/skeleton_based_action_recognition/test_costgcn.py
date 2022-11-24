# Copyright 2020-2022 OpenDR European Project
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

from opendr.perception.skeleton_based_action_recognition import CoSTGCNLearner
from opendr.engine.datasets import ExternalDataset
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

        # Download model weights
        # CoSTGCNLearner.download(path=Path(cls.temp_dir) / "weights", model_names={_BACKBONE})
        cls.learner = CoSTGCNLearner(
            device=device,
            temp_path=str(cls.temp_dir),
            iters=1,
            batch_size=2,
            backbone=_BACKBONE,
            num_workers=0,
        )

        # Download all required files for testing
        cls.pretrained_weights_path = cls.temp_dir / "weights" / "costgcn_ntu60_xview_joint.ckpt"
        # cls.pretrained_weights_path = cls.learner.download(
        #     path=os.path.join(cls.temp_dir, "pretrained_models", "costgcn"),
        #     method_name="costgcn",
        #     mode="pretrained",
        #     file_name="stgcn_ntu60_xview_joint.ckpt",
        # )
        cls.Train_DATASET_PATH = cls.learner.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data")
        )
        cls.Val_DATASET_PATH = cls.learner.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data")
        )

    # @classmethod
    # def tearDownClass(cls):
    #     try:
    #         shutil.rmtree(str(cls.temp_dir))
    #     except OSError as e:
    #         logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def xtest_fit(self):
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

    def xtest_eval(self):
        test_ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )

        self.learner.model.load_state_dict(
            self.learner.model.map_state_dict(
                torch.load(self.pretrained_weights_path, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
            ),
            strict=True,
        )
        results = self.learner.eval(test_ds, steps=2)

        assert results["accuracy"] > 0.5
        assert results["loss"] < 1

    def xtest_infer(self):
        ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))[0]
        frame = batch[:, :, -1]  # Select a single frame

        self.learner.model.clean_state()
        self.learner.model.forward_steps(batch[:, :, :-1])  # Init model state

        # Input is Tensor
        results1 = self.learner.infer(frame)
        # Results is a batch with each item summing to 1.0
        assert all([torch.isclose(torch.sum(r.confidence), torch.tensor(1.0)) for r in results1])

    def test_optimize(self):
        ds = self.learner._prepare_dataset(
            ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD"),
            data_filename="val_joints.npy",
            labels_filename="val_labels.pkl",
            skeleton_data_type="joint",
            phase="val",
            verbose=False,
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        it = iter(dl)
        batch = next(it)[0]

        self.learner.ort_session = None
        self.learner.load(self.temp_dir / "weights" / f"{_BACKBONE}_ntu60_xview_joint.ckpt")

        target = self.learner.infer(batch)

        self.learner.optimize()

        assert self.learner.ort_session is not None
        result = self.learner.infer(batch)

        # Clean up
        self.learner.ort_session = None

    def xtest_save_and_load(self):
        assert self.learner.model is not None
        self.learner.save(self.temp_dir)
        # Make changes to check subsequent load
        self.learner.model = None
        self.learner.batch_size = 42
        self.learner.load(self.temp_dir)
        self.assertIsNotNone(self.learner.model, "model is None after loading pth model.")
        assert self.learner.batch_size == 2


if __name__ == "__main__":
    unittest.main()
