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

import unittest
import cv2
import shutil
import os
import torch
from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner
from opendr.engine.datasets import ExternalDataset

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'


def rmfile(path):
    try:
        os.remove(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def rmdir(_dir):
    try:
        shutil.rmtree(_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


class TestBisenetLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST BisenetLearner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "binary_high_resolution")
        cls.learner = BinaryHighResolutionLearner(device=device)
        # Download CamVid dataset
        cls.dataset_path = os.path.join(".", "projects", "python", "perception", "binary_high_resolution",
                                        "demo_dataset")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_fit(self):
        dataset = ExternalDataset(self.dataset_path, "VOC2012")
        self.learner = BinaryHighResolutionLearner(device=device, iters=1)
        m = list(self.learner.model.parameters())[0].clone()
        self.learner.fit(dataset, silent=True)
        self.assertFalse(torch.equal(m, list(self.learner.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        dataset = ExternalDataset(self.dataset_path, "VOC2012")
        results = self.learner.eval(dataset)
        self.assertIsNotNone(results,
                             msg="Eval results dictionary not returned.")

    #
    def test_infer(self):
        img = cv2.imread(os.path.join(self.dataset_path, "test_img.png"))
        self.assertIsNotNone(self.learner.infer(img),
                             msg="Returned empty Heatmap.")

    def test_save_load_and_onnx(self):
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner.model = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.learner.model, "model is None after loading model.")

        self.learner.optimize()
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        img = cv2.imread(os.path.join(self.dataset_path, "test_img.png"))
        self.assertIsNotNone(self.learner.infer(img),
                             msg="Returned empty Heatmap.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))


if __name__ == "__main__":
    unittest.main()
