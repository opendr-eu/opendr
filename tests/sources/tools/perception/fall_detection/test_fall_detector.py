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
import shutil
import unittest

from opendr.engine.data import Image
from opendr.engine.datasets import ExternalDataset
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner

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


class TestFallDetectorLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Fall Detector Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "fall_detection",
                                    "fall_detector_temp")
        cls.pose_estimator = LightweightOpenPoseLearner(device=device, temp_path=cls.temp_dir,
                                                        mobilenet_use_stride=False)
        cls.pose_estimator.download(mode="pretrained")
        cls.pose_estimator.load(os.path.join(cls.temp_dir, "openpose_default"))

        cls.fall_detector = FallDetectorLearner(cls.pose_estimator)
        cls.fall_detector.download(path=cls.temp_dir, mode="test_data")

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "openpose_default"))
        rmdir(os.path.join(cls.temp_dir, "test_images"))
        rmdir(os.path.join(cls.temp_dir))

    def test_eval(self):
        eval_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "test_images"), dataset_type="test")

        results_dict = self.fall_detector.eval(eval_dataset)
        self.assertEqual(results_dict['accuracy'], 1.0,
                         msg="Accuracy is not 1.0.")
        self.assertEqual(results_dict['sensitivity'], 1.0,
                         msg="Sensitivity is not 1.0.")
        self.assertEqual(results_dict['specificity'], 1.0,
                         msg="Specificity is not 1.0.")
        self.assertEqual(results_dict['detection_accuracy'], 1.0,
                         msg="Detection accuracy is not 1.0.")
        self.assertEqual(results_dict['no_detections'], 0,
                         msg="Number of no detections is not 0.")

    def test_infer(self):
        img = Image.open(os.path.join(self.temp_dir, "test_images", "fallen.png"))
        # Detector should detect fallen person on fallen.png
        self.assertTrue(self.fall_detector.infer(img)[0][0].data == 1,
                        msg="Fall detector didn't detect fallen person on fallen.png")

        img = Image.open(os.path.join(self.temp_dir, "test_images", "standing.png"))
        # Detector should detect standing person on standing.png
        self.assertTrue(self.fall_detector.infer(img)[0][0].data == -1,
                        msg="Fall detector didn't detect standing person on standing.png")

        img = Image.open(os.path.join(self.temp_dir, "test_images", "no_person.png"))
        # Detector should not detect fallen nor standing person on no_person.png
        self.assertTrue(len(self.fall_detector.infer(img)) == 0,
                        msg="Fall detector detected fallen or standing person on no_person.png")
