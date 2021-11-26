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

import unittest
import os
import torch
import argparse
from opendr_internal.projects.data_generation.SyntheticDataGeneration import MultiviewDataGenerationLearner


class TestMultiviewDataGenerationLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Multiview Data Generation Learner\n"
              "**********************************")
        parser = argparse.ArgumentParser()
        try:
            if torch.cuda.is_available():
                print("GPU found.")
                parser.add_argument('-device', default='cuda', type=str, help='choose between cuda or cpu ')
            else:
                print("GPU not found. Using CPU instead.")
                parser.add_argument('-device', default='cpu', type=str, help='choose between cuda or cpu ')
        except:
            parser.add_argument("-device", default="cpu", type=str, help="choose between cuda or cpu ")
        parser.add_argument("-path_in", default=os.path.join("opendr_internal", "projects",
                                                             "data_generation",
                                                             "",
                                                             "demos", "imgs_input"),
                            type=str, help='Give the path of image folder')
        parser.add_argument('-path_3ddfa', default=os.path.join("opendr_internal", "projects",
                                                                "data_generation",
                                                                "",
                                                                "algorithm", "DDFA"),
                            type=str, help='Give the path of DDFA folder')
        parser.add_argument('-save_path', default=os.path.join("opendr_internal", "projects",
                                                               "data_generation",
                                                               "",
                                                               "results"),
                            type=str, help='Give the path of results folder')
        parser.add_argument('-val_yaw', default="10,20", nargs='+', type=str, help='yaw poses list between [-90,90] ')
        parser.add_argument('-val_pitch', default="30,40", nargs='+', type=str,
                            help='pitch poses list between [-90,90] ')
        args = parser.parse_args()

        cls.learner = MultiviewDataGenerationLearner(path_in=args.path_in, path_3ddfa=args.path_3ddfa, save_path=args.save_path,
                                                     val_yaw=args.val_yaw, val_pitch=args.val_pitch, device=args.device)

    def test_eval(self):

        self.learner.eval()
        DIR = os.path.join(os.environ['OPENDR_HOME'], "projects", "data_generation",
                           "", "results")

        # Default pretrained model extracts 4 rendered images
        self.assertAlmostEqual(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]), 4,
                               msg="The generated facial images must be more than 4 vertices.")


if __name__ == '__main__':
    unittest.main()
