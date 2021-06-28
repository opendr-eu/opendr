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
from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner

from PIL import Image

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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


class TestDetrLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join("tests", "sources", "tools",
                                    "perception", "object_detection_2d",
                                    "detr", "detr_temp")

        cls.model_backbone = "resnet50"

        cls.learner = DetrLearner(iters=1,
                                  temp_path=cls.temp_dir,
                                  backbone=cls.model_backbone,
                                  device=DEVICE)
        
        cls.learner.download()
        
        print("Model downloaded", file=sys.stderr)
        
        cls.learner.download(mode="test_data")
        
        print("Data downloaded", file=sys.stderr)
        
        cls.dataset_path = os.path.join(cls.temp_dir, "nano_coco")
        
        cls.dataset = ExternalDataset(
            cls.dataset_path, 
            "coco"
        )

        
    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, 'checkpoints'))
        rmdir(os.path.join(cls.temp_dir, 'facebookresearch_detr_master'))
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        self.learner.model = None
        self.learner.ort_session = None
        
        self.learner.fit(
            self.dataset,
            annotations_folder="", 
            train_annotations_file="instances.json", 
            train_images_folder="image", 
            verbose=True
        )

    def test_eval(self):
        self.learner.model = None
        self.learner.ort_session = None
        
        self.learner.download()
        
        result = self.learner.eval(
            self.dataset,
            images_folder='image',
            annotations_folder='',
            annotations_file='instances.json',
        )

        self.assertGreater(len(result), 0)

    def test_infer(self):
        self.learner.model = None
        self.learner.ort_session = None
        
        self.learner.download()
        
        image_path = os.path.join(
            self.dataset_path, 
            "image",
            "000000391895.jpg"
            )
        
        image = Image.open(image_path)
        
        result = self.learner.infer(image)
        
        self.assertGreater(len(result), 0)

    def test_save(self):
        self.learner.model = None
        self.learner.ort_session = None
        
        model_dir = os.path.join(self.temp_dir, "test_model")
        
        self.learner.download()
        
        self.learner.save(model_dir)
        
        starting_param_1 = list(self.learner.model.parameters())[0].clone()

        learner2 = DetrLearner(
            iters=1,
            temp_path=self.temp_dir,
            device=DEVICE,
        )
        learner2.load(model_dir)

        new_param = list(learner2.model.parameters())[0].clone()
        self.assertTrue(torch.equal(starting_param_1, new_param))
        
        rmdir(model_dir)
        
    def test_optimize(self):
        self.learner.model = None
        self.learner.ort_session = None
        
        self.learner.download()

        self.learner.optimize()


if __name__ == "__main__":
    unittest.main()
