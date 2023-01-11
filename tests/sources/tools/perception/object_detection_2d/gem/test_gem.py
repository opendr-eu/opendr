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
from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_detection_2d import GemLearner
import os
from PIL import Image

DEVICE = device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

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


class TestGemLearner(unittest.TestCase):
    temp_dir = os.path.join("tests",
                            "sources",
                            "tools",
                            "perception",
                            "object_detection_2d",
                            "gem",
                            "gem_temp",
                            )
    dataset_location = os.path.join(temp_dir, 'sample_dataset')
    learners = {}
    model_backbones = ["resnet50", "mobilenetv2"]

    @classmethod
    def setUpClass(cls):
        print("\n\n*********************************\nTEST Object Detection GEM Learner\n"
              "*********************************")
        for backbone in cls.model_backbones:
            cls.learners[backbone] = GemLearner(iters=1,
                                                temp_path=cls.temp_dir,
                                                backbone=backbone,
                                                num_classes=7,
                                                device=DEVICE,
                                                )

        for learner in cls.learners.values():
            learner.download(mode='pretrained_gem')

        print("Model downloaded", file=sys.stderr)

        cls.learners['resnet50'].download(mode='test_data_sample_dataset')

        cls.learners['resnet50'].download(mode='test_data_sample_images')

        print("Data downloaded", file=sys.stderr)
        cls.m1_dataset = ExternalDataset(cls.dataset_location, "coco")
        cls.m2_dataset = ExternalDataset(cls.dataset_location, "coco")

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, 'pretrained_models'))
        rmdir(os.path.join(cls.temp_dir, 'checkpoints'))
        rmdir(os.path.join(cls.temp_dir, 'facebookresearch_detr_master'))
        rmdir(os.path.join(cls.temp_dir, 'sample_dataset'))
        rmdir(os.path.join(cls.temp_dir, 'sample_images'))
        rmdir(os.path.join(cls.temp_dir, 'outputs'))
        rmdir(cls.temp_dir)

    def test_fit(self):
        # Test fit will issue resource warnings due to some files left open in pycoco tools,
        # as well as a deprecation warning due to a cast of a float to integer (hopefully they will be fixed in a future
        # version)
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        for backbone in self.model_backbones:
            self.learners[backbone].model = None
            self.learners[backbone].ort_session = None

            self.learners[backbone].download(mode='pretrained_gem')

            m = list(self.learners[backbone].model.parameters())[0].clone()

            self.learners[backbone].fit(m1_train_edataset=self.m1_dataset,
                                        m2_train_edataset=self.m2_dataset,
                                        annotations_folder='annotations',
                                        m1_train_annotations_file='RGB_26May2021_14h19m_coco.json',
                                        m2_train_annotations_file='Thermal_26May2021_14h19m_coco.json',
                                        m1_train_images_folder='train/m1',
                                        m2_train_images_folder='train/m2',
                                        out_dir=os.path.join(self.temp_dir, "outputs"),
                                        trial_dir=os.path.join(self.temp_dir, "trial"),
                                        logging_path='',
                                        verbose=False,
                                        m1_val_edataset=self.m1_dataset,
                                        m2_val_edataset=self.m2_dataset,
                                        m1_val_annotations_file='RGB_26May2021_14h19m_coco.json',
                                        m2_val_annotations_file='Thermal_26May2021_14h19m_coco.json',
                                        m1_val_images_folder='val/m1',
                                        m2_val_images_folder='val/m2',
                                        )

            self.assertFalse(torch.equal(m, list(self.learners[backbone].model.parameters())[0]),
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

        for backbone in self.model_backbones:
            self.learners[backbone].model = None
            self.learners[backbone].ort_session = None

            self.learners[backbone].download(mode='pretrained_gem')

            result = self.learners[backbone].eval(
                m1_edataset=self.m1_dataset,
                m2_edataset=self.m2_dataset,
                m1_images_folder='val/m1',
                m2_images_folder='val/m2',
                annotations_folder='annotations',
                m1_annotations_file='RGB_26May2021_14h19m_coco.json',
                m2_annotations_file='Thermal_26May2021_14h19m_coco.json',
                verbose=False,
            )

            self.assertGreater(len(result), 0)

        # Cleanup
        warnings.simplefilter("default", ResourceWarning)
        warnings.simplefilter("default", DeprecationWarning)

    def test_infer(self):
        m1_image = Image.open(os.path.join(self.temp_dir, "sample_images/rgb/2021_04_22_21_35_47_852516.jpg"))
        m2_image = Image.open(os.path.join(self.temp_dir, 'sample_images/aligned_infra/2021_04_22_21_35_47_852516.jpg'))

        for backbone in self.model_backbones:
            self.learners[backbone].model = None
            self.learners[backbone].ort_session = None
            self.learners[backbone].download(mode='pretrained_gem')
            result, _, _ = self.learners[backbone].infer(m1_image, m2_image)
            self.assertGreater(len(result), 0)

    def test_save(self):
        backbone = 'resnet50'
        self.learners[backbone].model = None
        self.learners[backbone].ort_session = None

        model_dir = os.path.join(self.temp_dir, "test_model")

        self.learners[backbone].download(mode='pretrained_detr')

        self.learners[backbone].save(model_dir)

        starting_param_1 = list(self.learners[backbone].model.parameters())[0].clone()

        learner2 = GemLearner(
            iters=1,
            temp_path=self.temp_dir,
            device=DEVICE,
            num_classes=7,
            backbone=backbone,
        )
        learner2.load(model_dir)

        new_param = list(learner2.model.parameters())[0].clone()
        self.assertTrue(torch.equal(starting_param_1, new_param))

        rmdir(model_dir)


if __name__ == "__main__":
    unittest.main()
