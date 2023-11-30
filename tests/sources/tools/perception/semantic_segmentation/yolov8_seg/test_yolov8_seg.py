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

from opendr.perception.semantic_segmentation import YOLOv8SegLearner

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


class TestYoloV8Seg(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST YOLOv8SegLearner\n"
              "**********************************")
        cls.learner = YOLOv8SegLearner("yolov8n-seg", device=device)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmfile(os.path.join(cls.learner.temp_path, "bus.jpg"))
        rmfile(os.path.join(cls.learner.temp_path, "yolov8n-seg.pt"))

    def test_infer(self):
        heatmap = self.learner.infer("https://ultralytics.com/images/bus.jpg", no_mismatch=True,
                                     verbose=True, image_size=480)
        self.assertEqual(heatmap.data.shape, (480, 384), msg="Invalid Heatmap shape.")
        self.assertEqual(heatmap.data[336][120], 80, msg="Unexpected Heatmap detection.")


if __name__ == "__main__":
    unittest.main()
