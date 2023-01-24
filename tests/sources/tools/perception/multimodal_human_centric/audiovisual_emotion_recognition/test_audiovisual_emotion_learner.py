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
import torch
import tempfile
import numpy as np
import random
import os

# OpenDR imports
from opendr.engine.data import Video, Timeseries
from opendr.engine.target import Category
from opendr.perception.multimodal_human_centric import AudiovisualEmotionLearner
from opendr.engine.datasets import DatasetIterator


DEVICE = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'


class DummyDataset(DatasetIterator):
    def __init__(self, n_class=8, n_sample=4):
        super(DummyDataset, self).__init__()
        self.n_sample = n_sample
        self.n_class = n_class

    def __len__(self,):
        return self.n_sample

    def __getitem__(self, i):
        xa = np.float32(np.random.rand(10, 156))
        xv = np.float32(np.random.rand(3, 15, 224, 224))
        y = np.random.randint(low=0, high=self.n_class)
        return Timeseries(xa), Video(xv), Category(y)


def get_random_learner():
    n_class = np.random.randint(low=2, high=10)

    fusion = random.choice(['it', 'ia', 'lt'])
    mod_drop = random.choice(['nodrop', 'noisedrop', 'zerodrop'])

    learner = AudiovisualEmotionLearner(num_class=n_class,
                                        iters=1,
                                        batch_size=2,
                                        fusion=fusion,
                                        mod_drop=mod_drop, device=DEVICE)
    return learner


class TestAudiovisualEmotionLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST AudiovisualEmotionLearner\n"
              "**********************************")
        pass

    @classmethod
    def tearDownClass(cls):
        return

    def test_fit(self):
        learner = get_random_learner()
        train_set = DummyDataset(learner.num_class)
        val_set = DummyDataset(learner.num_class)

        old_weight = list(learner.model.parameters())[0].clone()
        learner.fit(train_set, val_set, silent=True, verbose=False)
        new_weight = list(learner.model.parameters())[0].clone()

        self.assertFalse(torch.equal(old_weight, new_weight),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        learner = get_random_learner()
        dataset = DummyDataset(learner.num_class)
        performance = learner.eval(dataset, silent=True, verbose=False)

        self.assertTrue('cross_entropy' in performance.keys())
        self.assertTrue('acc' in performance.keys())

    def test_infer(self):
        temp_dir = tempfile.TemporaryDirectory()

        xa = Timeseries(np.float32(np.random.rand(10, 156)))
        xv = Video(np.float32(np.random.rand(3, 15, 224, 224)))

        # create learner and download pretrained model
        learner = AudiovisualEmotionLearner(num_class=8)

        # make inference
        pred = learner.infer(xa, xv)
        self.assertTrue(isinstance(pred, Category))

        self.assertTrue(pred.confidence <= 1,
                        msg="Confidence of prediction must be less or equal than 1")
        temp_dir.cleanup()

    def test_save_load(self):
        temp_dir = tempfile.TemporaryDirectory()
        learner = get_random_learner()

        learner.save(temp_dir.name, verbose=False)

        new_learner = AudiovisualEmotionLearner(
            num_class=learner.num_class, fusion=learner.fusion, mod_drop=learner.mod_drop)

        new_learner.load(temp_dir.name, verbose=False)
        xa = Timeseries(np.float32(np.random.rand(10, 156)))
        xv = Video(np.float32(np.random.rand(3, 15, 224, 224)))
        old_pred = learner.infer(xa, xv).confidence
        new_pred = new_learner.infer(xa, xv).confidence

        self.assertEqual(old_pred, new_pred)
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
