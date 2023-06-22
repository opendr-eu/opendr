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
from opendr.engine.data import Image
import shutil
import os
from opendr.simulation.human_model_generation import PIFuGeneratorLearner


def rmdir(_dir):
    try:
        shutil.rmtree(_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


class TestPIFuGeneratorLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST PIFu Generator Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(os.environ['OPENDR_HOME'], "tests", "sources", "tools", "simulation",
                                    "human_model_generation", "temp")
        cls.learner = PIFuGeneratorLearner(device='cuda', checkpoint_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir))

    def test_infer(self):

        img_rgb = Image.open(os.path.join(os.environ['OPENDR_HOME'], "projects", "python", "simulation",
                                          "human_model_generation", "demos", "imgs_input", "rgb", "result_0004.jpg"))
        img_msk = Image.open(os.path.join(os.environ['OPENDR_HOME'], "projects", "python", "simulation",
                                          "human_model_generation", "demos", "imgs_input", "msk", "result_0004.jpg"))
        model_3D = self.learner.infer(imgs_rgb=[img_rgb], imgs_msk=[img_msk], extract_pose=False)

        # Default pretrained mobilenet model detects 18 keypoints on img with id 785
        self.assertGreater(model_3D.get_vertices().shape[0], 52260,
                           msg="The generated 3D must have more than 52260 vertices.")


if __name__ == "__main__":
    unittest.main()
