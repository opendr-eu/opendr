# Copyright 1996-2020 OpenDR European Project
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
import cv2
import shutil
import os
import torch
from perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import LightweightOpenPoseLearner
from engine.data import Image
from engine.datasets import ExternalDataset


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
    temp_dir = "." + os.sep + "tests" + os.sep + "sources" + os.sep + "perception" + os.sep + "pose_estimation" + \
               os.sep + "lightweight_open_pose"

    # TODO download all local files needed and clean them up afterwards

    def test_fit(self):
        pose_estimator = LightweightOpenPoseLearner(device="cpu", temp_path=self.temp_dir, batch_size=1, epochs=1)
        training_dataset = ExternalDataset(path=self.temp_dir + os.sep + "dataset", dataset_type="COCO")
        pose_estimator.init_model()
        m = list(pose_estimator.model.parameters())[0].clone()
        # TODO check for results return
        pose_estimator.fit(dataset=training_dataset, silent=True,
                           images_folder_name="image", annotations_filename="annotation.json")
        self.assertFalse(torch.equal(m, list(pose_estimator.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        pass # TODO this

    def test_infer(self):
        pose_estimator = LightweightOpenPoseLearner(device="cpu")
        pose_estimator.load(self.temp_dir + os.sep + "trainedModel")
        img = cv2.imread(self.temp_dir + os.sep + "dataset" + os.sep + "image" + os.sep + "000000000785.jpg")
        # Default pretrained mobilenet model detects 18 keypoints on img with id 785
        self.assertGreater(len(pose_estimator.infer(img)[0].data), 0,
                           msg="Returned pose must have non-zero number of keypoints.")
        # TODO cleanup downloaded trainedModel

    def test_save_load(self):
        pose_estimator = LightweightOpenPoseLearner(device="cpu", temp_path=self.temp_dir)
        pose_estimator.init_model()
        pose_estimator.save(self.temp_dir + os.sep + "testModel")
        pose_estimator.model = None
        pose_estimator.load(self.temp_dir + os.sep + "testModel")
        self.assertIsNotNone(pose_estimator.model, "model is None after loading pth model.")
        rmdir(self.temp_dir + os.sep + "testModel")

    def test_save_load_onnx(self):
        pose_estimator = LightweightOpenPoseLearner(device="cpu", temp_path=self.temp_dir)
        pose_estimator.init_model()
        pose_estimator.optimize()
        pose_estimator.save(self.temp_dir + os.sep + "testModel")
        pose_estimator.model = None
        pose_estimator.load(self.temp_dir + os.sep + "testModel")
        self.assertIsNotNone(pose_estimator.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        rmfile(self.temp_dir + os.sep + "onnx_model_temp.onnx")
        rmdir(self.temp_dir + os.sep + "testModel")

    def test_optimize(self):
        pose_estimator = LightweightOpenPoseLearner(device="cpu", temp_path=self.temp_dir)
        pose_estimator.load(self.temp_dir + os.sep + "trainedModel")
        pose_estimator.optimize()
        self.assertIsNotNone(pose_estimator.ort_session)
        # Cleanup
        rmfile(self.temp_dir + os.sep + "onnx_model_temp.onnx")


if __name__ == "__main__":
    unittest.main()
