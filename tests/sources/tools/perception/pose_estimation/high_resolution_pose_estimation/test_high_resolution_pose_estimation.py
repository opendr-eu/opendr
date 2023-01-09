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
import shutil
from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner

from opendr.engine.datasets import ExternalDataset
from opendr.engine.data import Image
import warnings
import os

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


class TestLightweightOpenPoseLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST High Resolution Pose Estimation Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception",
                                    "pose_estimation", "high_resolution_pose_estimation", "hr_pose_estim_temp")
        cls.pose_estimator = HighResolutionPoseEstimationLearner(device=device, temp_path=cls.temp_dir,  num_workers=1)

        # Download all required files for testing
        cls.pose_estimator.download(mode="pretrained")
        cls.pose_estimator.download(mode="test_data")

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "openpose_default"))
        rmdir(os.path.join(cls.temp_dir, "dataset"))

        rmdir(os.path.join(cls.temp_dir))

    def test_eval(self):
        # Test eval will issue resource warnings due to some files left open in pycoco tools,
        # as well as a deprecation warning due to a cast of a float to integer (hopefully they will be fixed in a future
        # version)
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        eval_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "dataset"), dataset_type="COCO")
        self.pose_estimator.load(os.path.join(self.temp_dir, "openpose_default"))
        results_dict = self.pose_estimator.eval(eval_dataset, use_subset=False, verbose=True, silent=True,
                                                images_folder_name="image", annotations_filename="annotation.json")
        self.assertNotEqual(len(results_dict['average_precision']), 0,
                            msg="Eval results dictionary contains empty list.")
        self.assertNotEqual(len(results_dict['average_recall']), 0,
                            msg="Eval results dictionary contains empty list.")
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "detections.json"))
        warnings.simplefilter("default", ResourceWarning)
        warnings.simplefilter("default", DeprecationWarning)

    def test_infer(self):
        self.pose_estimator.model = None
        self.pose_estimator.load(os.path.join(self.temp_dir, "openpose_default"))

        img = Image.open(os.path.join(self.temp_dir, "dataset", "image", "000000000785_1080.jpg"))
        # Default pretrained mobilenet model detects 18 keypoints on img with id 785
        self.assertGreater(len(self.pose_estimator.infer(img)[0].data), 0,
                           msg="Returned pose must have non-zero number of keypoints.")


if __name__ == "__main__":
    unittest.main()
