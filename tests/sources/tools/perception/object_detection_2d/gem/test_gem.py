# Copyright 2020-2021 OpenDR European Project
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

import sys
import unittest
import shutil
import os
import torch

from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_detection_2d.gem.gem_learner import GEMLearner

from PIL import Image

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

print("Using device:", DEVICE)
print("Using device:", DEVICE, file=sys.stderr)


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


class TestGEMLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join("tests", "sources", "tools",
                                    "perception", "object_detection_2d",
                                    "gem", "gem_temp")

        cls.model_backbone = "resnet50"

        cls.learner = GEMLearner(iters=1,
                                  temp_path=cls.temp_dir,
                                  backbone=cls.model_backbone,
                                  num_classes=7,
                                  device=DEVICE)

        cls.learner.create_model(pretrained='gem_l515')

        print("Model downloaded", file=sys.stderr)

        cls.learner.download_l515()

        print("Data downloaded", file=sys.stderr)
        dataset_location = os.path.join(cls.learner.datasetargs.dataset_root, \
                            cls.learner.datasetargs.dataset_name)
        cls.m1_dataset = ExternalDataset(
            dataset_location,
            "coco"
        )
        cls.m2_dataset = ExternalDataset(
            dataset_location,
            "coco"
        )


    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir('./pretrained_models')
        rmdir('./datasets')
        rmdir('./outputs')
        rmdir(cls.temp_dir)

    def test_fit(self):
        self.learner.model = None
        self.learner.ort_session = None

        self.learner.fit(
            verbose=True
        )

    def test_eval(self):
        self.learner.model = None
        self.learner.ort_session = None

        self.learner.create_model(pretrained='gem_l515')

        result = self.learner.eval()

        self.assertGreater(len(result), 0)

    def test_infer(self):
        self.learner.model = None
        self.learner.ort_session = None

        self.learner.create_model(pretrained='gem_l515')

        m1_image = Image.open("./datasets/l515_dataset/rgb/2021_04_22_21_35_47_852516.jpg")
        m2_image = Image.open("./datasets/l515_dataset/infra_aligned/2021_04_22_21_35_47_852516.jpg")

        result = self.learner.infer(m1_image, m2_image)

        self.assertGreater(len(result), 0)

    def test_save(self):
        self.learner.model = None
        self.learner.ort_session = None

        model_dir = os.path.join(self.temp_dir, "test_model")

        self.learner.create_model(pretrained='detr_coco')

        self.learner.save(model_dir)

        starting_param_1 = list(self.learner.model.parameters())[0].clone()

        learner2 = GEMLearner(
            iters=1,
            temp_path=self.temp_dir,
            device=DEVICE,
            num_classes=7,
        )
        learner2.load(model_dir)

        new_param = list(learner2.model.parameters())[0].clone()
        self.assertTrue(torch.equal(starting_param_1, new_param))

        rmdir(model_dir)


if __name__ == "__main__":
    unittest.main()
