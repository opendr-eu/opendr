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
from opendr.perception.semantic_segmentation import BisenetLearner
from opendr.perception.semantic_segmentation import CamVidDataset


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

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "semantic_segmentation",
                                    "bisenet", "temp")
        cls.learner = BisenetLearner()
        # Download all required files for testing
        cls.learner.download(path=os.path.join(cls.temp_dir, "bisenet_camvid"), mode="pretrained")
        cls.learner.download(path=cls.temp_dir, mode="testingImage")

        # Download CamVid dataset
        cls.dataset_path = os.path.join(cls.temp_dir, "datasets")
        CamVidDataset.download_data(os.path.join(cls.temp_dir, "datasets"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "test1.png"))
        rmdir(os.path.join(cls.temp_dir, "bisenet_camvid"))
        rmdir(cls.temp_dir)
        pass

    def test_fit(self):
        training_dataset = CamVidDataset(os.path.join(self.dataset_path, "CamVid"), mode='train')
        m = list(self.learner.model.parameters())[0].clone()
        self.learner.fit(dataset=training_dataset, silent=True)
        self.assertFalse(torch.equal(m, list(self.learner.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        eval_dataset = CamVidDataset(os.path.join(self.dataset_path, "CamVid"), mode='test')
        self.learner.load(path=os.path.join(self.temp_dir, "bisenet_camvid"))
        results = self.learner.eval(eval_dataset)
        self.assertIsNotNone(results,
                             msg="Eval results dictionary not returned.")

    def test_infer(self):
        self.learner.load(os.path.join(self.temp_dir, "bisenet_camvid"))
        img = cv2.imread(os.path.join(self.temp_dir, "test1.png"))
        self.assertIsNotNone(self.learner.infer(img),
                             msg="Returned empty Heatmap.")

    def test_save_load(self):
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner.model = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.learner.model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))


if __name__ == "__main__":
    unittest.main()
