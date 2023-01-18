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
import os
import torch
import tempfile
import numpy as np
import random
import imageio
from urllib.request import urlretrieve
import cv2

# OpenDR imports
from opendr.perception.multimodal_human_centric import RgbdHandGestureLearner, get_builtin_architectures
from opendr.engine.datasets import DatasetIterator
from opendr.engine.data import Image
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL


class DummyDataset(DatasetIterator):
    def __init__(self, n_class, n_sample=4):
        super(DummyDataset, self).__init__()
        self.n_sample = n_sample
        self.n_class = n_class

    def __len__(self,):
        return self.n_sample

    def __getitem__(self, i):
        x = np.float32(np.random.rand(224, 224, 4))
        y = np.random.randint(low=0, high=self.n_class)
        return Image(x), Category(y)


def get_random_learner():
    n_class = np.random.randint(low=2, high=10)
    architecture = random.choice(get_builtin_architectures())

    learner = RgbdHandGestureLearner(n_class=n_class,
                                     architecture=architecture,
                                     iters=1,
                                     batch_size=2,
                                     test_mode=True,
                                     pretrained=False)
    return learner


class TestRgbdHandGestureLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST RgbdHandGestureLearner\n"
              "**********************************")
        pass

    @classmethod
    def tearDownClass(cls):
        return

    def test_fit(self):
        learner = get_random_learner()
        train_set = DummyDataset(learner.n_class)
        val_set = DummyDataset(learner.n_class)
        test_set = DummyDataset(learner.n_class)

        old_weight = list(learner.model.parameters())[0].clone()
        learner.fit(train_set, val_set, test_set, silent=True, verbose=False)
        new_weight = list(learner.model.parameters())[0].clone()

        self.assertFalse(torch.equal(old_weight, new_weight),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        learner = get_random_learner()
        dataset = DummyDataset(learner.n_class)
        performance = learner.eval(dataset, silent=True, verbose=False)

        self.assertTrue('cross_entropy' in performance.keys())
        self.assertTrue('acc' in performance.keys())

    def test_infer(self):
        temp_dir = tempfile.TemporaryDirectory()
        rgb_url = os.path.join(OPENDR_SERVER_URL,
                               'perception',
                               'multimodal_human_centric',
                               'rgbd_hand_gesture_learner',
                               'test_hand_gesture_rgb.png')
        depth_url = os.path.join(OPENDR_SERVER_URL,
                                 'perception',
                                 'multimodal_human_centric',
                                 'rgbd_hand_gesture_learner',
                                 'test_hand_gesture_depth.png')
        # retrieve test files
        rgb_file = os.path.join(temp_dir.name, 'rgb.png')
        depth_file = os.path.join(temp_dir.name, 'depth.png')
        urlretrieve(rgb_url, rgb_file)
        urlretrieve(depth_url, depth_file)

        # load test files
        rgb_img = np.asarray(imageio.imread(rgb_file)) / 255.0
        rgb_img = cv2.resize(rgb_img, (224, 224))
        depth_img = np.asarray(imageio.imread(depth_file)) / 65535.0
        depth_img = cv2.resize(depth_img, (224, 224))
        depth_img = np.expand_dims(depth_img, axis=-1)
        img = np.concatenate([rgb_img, depth_img], axis=-1)

        # normalize
        mean = np.asarray([0.485, 0.456, 0.406, 0.0303]).reshape(1, 1, 4)
        std = np.asarray([0.229, 0.224, 0.225, 0.0353]).reshape(1, 1, 4)
        img = (img - mean) / std
        img = Image(img, np.float32)

        # create learner and download pretrained model
        learner = RgbdHandGestureLearner(n_class=16, architecture='mobilenet_v2')
        model_path = os.path.join(temp_dir.name, 'mobilenet_v2')
        learner.download(model_path)
        learner.load(model_path)

        # make inference
        pred = learner.infer(img)
        self.assertTrue(isinstance(pred, Category))
        self.assertTrue(pred.data == 12,
                        msg="Predicted class label must be 12")
        self.assertTrue(pred.confidence <= 1,
                        msg="Confidence of prediction must be less or equal than 1")
        temp_dir.cleanup()

    def test_save_load(self):
        temp_dir = tempfile.TemporaryDirectory()
        learner = get_random_learner()

        learner.save(temp_dir.name, verbose=False)

        new_learner = RgbdHandGestureLearner(n_class=learner.n_class,
                                             architecture=learner.architecture)

        new_learner.load(temp_dir.name, verbose=False)
        img = Image(np.random.rand(224, 224, 4))
        old_pred = learner.infer(img).confidence
        new_pred = new_learner.infer(img).confidence

        self.assertEqual(old_pred, new_pred)
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
