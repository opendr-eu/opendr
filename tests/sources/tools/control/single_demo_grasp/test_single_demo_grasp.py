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

import shutil
import unittest
import torch

# detectron dependencies
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# OpenDR dependencies
from opendr.control.single_demo_grasp import SingleDemoGraspLearner
from opendr.engine.data import Image
import os

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'
# variable definitions here
dir_temp = os.path.join(".", "tests", "sources", "tools", "control", "single_demo_grasp", "sdg_temp")


def load_old_weights():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)

    return list(model.parameters())[0].clone()


def load_weights_from_file(path_to_model):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = path_to_model  # check if it's necessary
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)
    DetectionCheckpointer(model).load(path_to_model)

    return list(model.parameters())[0].clone()


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


class TestSingleDemoGraspLearner(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST SingleDemoGrasp Learner\n"
              "**********************************")
        cls.learner = SingleDemoGraspLearner(object_name='pendulum', data_directory=dir_temp, lr=0.0008, batch_size=1,
                                             num_workers=2, num_classes=1, iters=10, threshold=0.8, device=device,
                                             img_per_step=2)

        # Download all required files for testing
        cls.learner.download(path=dir_temp, object_name="pendulum")
        shutil.copy(os.path.join(cls.learner.output_dir, "model_final.pth"), os.path.join(cls.learner.output_dir,
                                                                                          "pretrained.pth"))

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for SingleDemoGrasp...')
        # Clean up downloaded files
        rmdir(dir_temp)
        del cls.learner

    def test_fit(self):
        print('Starting training test for SingleDemoGrasp...')
        self.learner.fit()
        old_weights = load_old_weights()
        new_weights = load_weights_from_file(os.path.join(self.learner.output_dir, "model_final.pth"))

        self.assertFalse(torch.equal(old_weights, new_weights),
                         msg="Fit method did not alter model weights")

    def test_infer(self):

        print('Starting inference test for SingleDemoGrasp...')
        sample_image = Image.open(os.path.join(dir_temp, "pendulum", "images", "val", "0.jpg"))
        self.learner.load(os.path.join(self.learner.output_dir, "pretrained.pth"))

        flag, _, _ = self.learner.infer(sample_image)
        self.assertTrue(flag == 1, msg="predictions are available with confidence more than threshold")

    def test_save_load(self):
        """
         learner load function only sets path to where you store the weights
         so we check whether the saved and loaded files after running these functions
         to make sure the files are correctly updated in the directory
        """

        print('Starting save_load test for SingleDemoGrasp...')
        self.learner.save(dir_temp)
        self.learner.load(os.path.join(dir_temp, self.learner.object_name, "output", "model_final.pth"))

        old_weights = load_old_weights()
        new_weights = load_weights_from_file(os.path.join(dir_temp, "model_final.pth"))

        self.assertFalse(torch.equal(old_weights, new_weights),
                         msg="load method did not alter model weights")

        print("Finished tesing save/load functions.")


if __name__ == "__main__":
    unittest.main()
