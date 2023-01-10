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

import sys
import unittest
import shutil
import torch
import warnings
from torch.jit import TracerWarning
from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_detection_2d import DetrLearner
from PIL import Image
import os

DEVICE = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

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
        print("\n\n**********************************\nTEST Object Detection DETR Learner\n"
              "**********************************")
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
        rmdir(os.path.join(cls.temp_dir, 'detr_default/checkpoints'))
        rmdir(os.path.join(cls.temp_dir, 'detr_default/facebookresearch_detr_master'))
        rmdir(os.path.join(cls.temp_dir, 'detr_default'))
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        # Test fit will issue resource warnings due to some files left open in pycoco tools,
        # as well as a deprecation warning due to a cast of a float to integer (hopefully they will be fixed in a future
        # version)
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.download(verbose=False)

        m = list(self.learner.model.parameters())[0].clone()

        self.learner.fit(
            self.dataset,
            annotations_folder="",
            train_annotations_file="instances.json",
            train_images_folder="image",
            silent=True
        )

        self.assertFalse(torch.equal(m, list(self.learner.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

        # Cleanup
        warnings.simplefilter("default", ResourceWarning)
        warnings.simplefilter("default", DeprecationWarning)

    def test_eval(self):
        # Test eval will issue resource warnings due to some files left open in pycoco tools,
        # as well as a deprecation warning due to a cast of a float to integer (hopefully they will be fixed in a future
        # version)
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        self.learner.model = None
        self.learner.ort_session = None

        self.learner.download(verbose=False)

        results_dict = self.learner.eval(
            self.dataset,
            images_folder='image',
            annotations_folder='',
            annotations_file='instances.json',
            verbose=False,
        )

        self.assertNotEqual(len(results_dict), 0,
                            msg="Eval results dictionary contains empty list.")
        # Cleanup
        warnings.simplefilter("default", ResourceWarning)
        warnings.simplefilter("default", DeprecationWarning)

    def test_infer(self):
        self.learner.model = None
        self.learner.ort_session = None

        self.learner.download(verbose=False)

        image_path = os.path.join(
            self.dataset_path,
            "image",
            "000000391895.jpg",
        )

        image = Image.open(image_path)

        result = self.learner.infer(image)

        self.assertGreater(len(result), 0)

    def test_save(self):
        self.learner.model = None
        self.learner.ort_session = None

        model_dir = os.path.join(self.temp_dir, "test_model")

        self.learner.download(verbose=False)

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

    def test_save_load(self):
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.download(verbose=False)
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner.model = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.learner.model, "model is None after loading pth model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))

    def test_save_load_onnx(self):
        # ONNX will issue TracerWarnings, but these can be ignored safely
        # because we use this function to create tensors out of constant
        # variables that are the same every time we call this function.
        warnings.simplefilter("ignore",  TracerWarning)
        warnings.simplefilter("ignore",  RuntimeWarning)

        self.learner.model = None
        self.learner.ort_session = None
        self.learner.download(verbose=False)
        self.learner.optimize()
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner.model = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.learner.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))
        rmdir(os.path.join(self.temp_dir, "test_model"))
        warnings.simplefilter("default",  TracerWarning)
        warnings.simplefilter("default",  RuntimeWarning)

    def test_optimize(self):
        # ONNX will issue TracerWarnings, but these can be ignored safely
        # because we use this function to create tensors out of constant
        # variables that are the same every time we call this function.
        warnings.simplefilter("ignore",  TracerWarning)
        warnings.simplefilter("ignore",  RuntimeWarning)

        self.learner.model = None
        self.learner.ort_session = None

        self.learner.download(verbose=False)

        self.learner.optimize()
        self.assertIsNotNone(self.learner.ort_session)
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))
        warnings.simplefilter("default",  TracerWarning)
        warnings.simplefilter("default",  RuntimeWarning)


if __name__ == "__main__":
    unittest.main()
