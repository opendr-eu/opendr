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
from opendr.perception.object_tracking_3d.ab3dmot.object_tracking_3d_ab3dmot_learner import (
    ObjectTracking3DAb3dmotLearner
)
from opendr.perception.object_tracking_3d.datasets.kitti_tracking import KittiTrackingDatasetIterator


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


class TestObjectTracking3DAb3dmot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Object Tracking 3D ab3dmot Learner\n"
              "**********************************")
        cls.temp_dir = os.path.join("tests", "sources", "tools",
                                    "perception", "object_tracking_3d",
                                    "ab3dmot",
                                    "ab3dmot_temp")

        cls.dataset = KittiTrackingDatasetIterator.download_labels(
            cls.temp_dir, True
        )

        print("Dataset downloaded", file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files

        rmdir(os.path.join(cls.temp_dir))

    def test_unsupported(self):

        learner = ObjectTracking3DAb3dmotLearner()

        with self.assertRaises(NotImplementedError):
            learner.save(None)
        with self.assertRaises(NotImplementedError):
            learner.load(None)
        with self.assertRaises(NotImplementedError):
            learner.fit(None)

    def test_eval(self):

        learner = ObjectTracking3DAb3dmotLearner()
        results = learner.eval(self.dataset, count=1)

        self.assertTrue("car" in results)
        self.assertTrue("pedestrian" in results)
        self.assertTrue("cyclist" in results)

    def test_infer(self):

        learner = ObjectTracking3DAb3dmotLearner()
        result = learner.infer(self.dataset[0][0][:5])

        # 5 input images are given, so 5 non-empty outputs should be returned
        self.assertTrue(len(result) == 5)
        self.assertTrue(len(result[0]) > 0)


if __name__ == "__main__":
    unittest.main()
