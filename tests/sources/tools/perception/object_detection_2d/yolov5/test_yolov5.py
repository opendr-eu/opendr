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
import gc
import cv2
import shutil
import os
import torch

from opendr.perception.object_detection_2d import YOLOv5DetectorLearner

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # workaround for rate limit bug
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


class TestYOLOv5DetectorLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST YOLOv5Detector Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "yolov5", "yolov5_temp")
        cls.detector = YOLOv5DetectorLearner(model_name='yolov5s', device=device, temp_path=cls.temp_dir,
                                             force_reload=True)

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for YOLOv5...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "zidane.jpg"))
        rmfile(os.path.join(cls.temp_dir, "yolov5s.pt"))
        rmdir(os.path.join(cls.temp_dir))

        del cls.detector
        gc.collect()
        print('Finished cleaning for YOLOv5...')

    def test_infer(self):
        print('Starting inference test for YOLOv5...')
        torch.hub.download_url_to_file('https://ultralytics.com/images/zidane.jpg', os.path.join(self.temp_dir, 'zidane.jpg'))
        img = cv2.imread(os.path.join(self.temp_dir, "zidane.jpg"))
        self.assertIsNotNone(self.detector.infer(img),
                             msg="Returned empty BoundingBoxList.")
        del img
        gc.collect()
        print('Finished inference test for YOLOv5...')


if __name__ == "__main__":
    unittest.main()
