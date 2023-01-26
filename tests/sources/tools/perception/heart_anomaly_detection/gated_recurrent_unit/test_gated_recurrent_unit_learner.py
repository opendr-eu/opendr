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

# OpenDR imports
from opendr.perception.heart_anomaly_detection import GatedRecurrentUnitLearner, get_AF_dataset
from opendr.engine.datasets import DatasetIterator
from opendr.engine.data import Timeseries
from opendr.engine.target import Category


class DummyDataset(DatasetIterator):
    def __init__(self, in_channels, series_length, n_class, n_sample=4):
        super(DummyDataset, self).__init__()
        self.in_channels = in_channels
        self.series_length = series_length
        self.n_sample = n_sample
        self.n_class = n_class

    def __len__(self,):
        return self.n_sample

    def __getitem__(self, i):
        x = np.random.rand(self.in_channels, self.series_length)
        y = np.random.randint(low=0, high=self.n_class)
        return Timeseries(x), Category(y)


class TestGatedRecurrentUnitLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Gated Recurrent Unit Learner\n"
              "**********************************")
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_file = os.path.join(cls.temp_dir.name, 'data.pickle')
        pass

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        return

    def test_fit(self):
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=10)
        recurrent_unit = 10

        train_set = DummyDataset(in_channels, series_length, n_class)
        val_set = DummyDataset(in_channels, series_length, n_class)
        test_set = DummyDataset(in_channels, series_length, n_class)

        learner = GatedRecurrentUnitLearner(in_channels=in_channels,
                                            series_length=series_length,
                                            n_class=n_class,
                                            recurrent_unit=recurrent_unit,
                                            iters=1,
                                            batch_size=4,
                                            test_mode=True)

        old_weight = list(learner.model.parameters())[0].clone()
        learner.fit(train_set, val_set, test_set, silent=True, verbose=False)
        new_weight = list(learner.model.parameters())[0].clone()

        self.assertFalse(torch.equal(old_weight, new_weight),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=100)
        recurrent_unit = 32

        learner = GatedRecurrentUnitLearner(in_channels=in_channels,
                                            series_length=series_length,
                                            n_class=n_class,
                                            recurrent_unit=recurrent_unit,
                                            iters=1,
                                            batch_size=4,
                                            test_mode=True)

        dataset = DummyDataset(in_channels, series_length, n_class)
        performance = learner.eval(dataset, silent=True, verbose=False)

        self.assertTrue('cross_entropy' in performance.keys())
        self.assertTrue('acc' in performance.keys())
        self.assertTrue('precision' in performance.keys())
        self.assertTrue('recall' in performance.keys())
        self.assertTrue('f1' in performance.keys())

    def test_infer(self):
        in_channels = 1
        series_length = 9000
        sample_length = 30
        n_class = 4
        recurrent_unit = 256
        fold_idx = 0

        learner = GatedRecurrentUnitLearner(in_channels=in_channels,
                                            series_length=series_length,
                                            n_class=n_class,
                                            recurrent_unit=recurrent_unit,
                                            iters=1,
                                            batch_size=4,
                                            test_mode=True)

        temp_dir = tempfile.TemporaryDirectory()
        learner.download(temp_dir.name, fold_idx)
        learner.load(temp_dir.name)

        train_set = get_AF_dataset(self.data_file, fold_idx, sample_length)[0]
        series = train_set[0][0]

        pred = learner.infer(series)
        self.assertTrue(isinstance(pred, Category))
        self.assertTrue(pred.data == 0,
                        msg="Predicted class label must be 0")
        self.assertTrue(pred.confidence <= 1,
                        msg="Confidence of prediction must be less or equal than 1")
        temp_dir.cleanup()

    def test_save_load(self):
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=100)
        recurrent_unit = 32

        learner = GatedRecurrentUnitLearner(in_channels=in_channels,
                                            series_length=series_length,
                                            n_class=n_class,
                                            recurrent_unit=recurrent_unit,
                                            iters=1,
                                            batch_size=4,
                                            test_mode=True)
        temp_dir = tempfile.TemporaryDirectory()
        learner.save(temp_dir.name, verbose=False)

        new_learner = GatedRecurrentUnitLearner(in_channels=in_channels,
                                                series_length=series_length,
                                                n_class=n_class,
                                                recurrent_unit=recurrent_unit,
                                                iters=1,
                                                batch_size=4,
                                                test_mode=True)

        new_learner.load(temp_dir.name, verbose=False)
        series = Timeseries(np.random.rand(in_channels, series_length))
        old_pred = learner.infer(series).confidence
        new_pred = new_learner.infer(series).confidence

        self.assertEqual(old_pred, new_pred)
        temp_dir.cleanup()

    def test_get_AF_dataset(self):
        fold_idx = random.choice([0, 1, 2, 3, 4])
        sample_length = random.choice(range(30, 60))
        train_set, val_set, series_length, class_weight = get_AF_dataset(self.data_file, fold_idx, sample_length)

        self.assertTrue(isinstance(train_set, DatasetIterator),
                        msg='`train_set` must be an instance of DatasetIterator')
        self.assertTrue(isinstance(val_set, DatasetIterator),
                        msg='`val_set` must be an instance of DatasetIterator')

        x, y = train_set[0]
        self.assertTrue(isinstance(x, Timeseries),
                        msg='first element generated by dataset must be an instance of Timeseries')
        self.assertTrue(isinstance(y, Category),
                        msg='second element generated by dataset must be an instance of Category')

    def test_download(self):
        series_length = 30 * 300
        recurrent_unit = random.choice([256, 512])
        fold_idx = random.choice([0, 1, 2, 3, 4])
        learner = GatedRecurrentUnitLearner(in_channels=1,
                                            series_length=series_length,
                                            n_class=4,
                                            recurrent_unit=recurrent_unit)

        temp_dir = tempfile.TemporaryDirectory()
        learner.download(temp_dir.name, fold_idx)

        metadata_file = os.path.join(temp_dir.name, 'metadata.json')
        weights_file = os.path.join(temp_dir.name, 'model_weights.pt')
        self.assertTrue(os.path.exists(metadata_file),
                        msg='metadata file was not downloaded')
        self.assertTrue(os.path.exists(weights_file),
                        msg='model weights file was not downloaded')

        learner.load(temp_dir.name)
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
