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

# OpenDR imports
from opendr.perception.heart_anomaly_detection import AttentionNeuralBagOfFeatureLearner
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


class TestAttentionNeuralBagOfFeatureLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Attention Neural Bag-of-Feature Learner\n"
              "**********************************")
        pass

    @classmethod
    def tearDownClass(cls):
        return

    def test_fit(self):
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=100)
        quantization_type = random.choice(['nbof', 'tnbof'])
        attention_type = random.choice(['spatial', 'temporal', 'spatialsa', 'temporalsa', 'spatiotemporal'])

        train_set = DummyDataset(in_channels, series_length, n_class)
        val_set = DummyDataset(in_channels, series_length, n_class)
        test_set = DummyDataset(in_channels, series_length, n_class)

        learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                     series_length,
                                                     n_class,
                                                     quantization_type=quantization_type,
                                                     attention_type=attention_type,
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
        quantization_type = random.choice(['nbof', 'tnbof'])
        attention_type = random.choice(['spatial', 'temporal', 'spatialsa', 'temporalsa', 'spatiotemporal'])

        learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                     series_length,
                                                     n_class,
                                                     quantization_type=quantization_type,
                                                     attention_type=attention_type,
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
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=100)
        quantization_type = random.choice(['nbof', 'tnbof'])
        attention_type = random.choice(['spatial', 'temporal', 'spatialsa', 'temporalsa', 'spatiotemporal'])

        learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                     series_length,
                                                     n_class,
                                                     quantization_type=quantization_type,
                                                     attention_type=attention_type,
                                                     iters=1,
                                                     batch_size=4,
                                                     test_mode=True)

        series = Timeseries(np.random.rand(in_channels, series_length))
        pred = learner.infer(series)
        self.assertTrue(isinstance(pred, Category))
        self.assertTrue(pred.data < learner.n_class,
                        msg="Predicted class label must be less than the number of class")

    def test_save_load(self):
        in_channels = random.choice([1, 2])
        series_length = random.choice([30 * 300, 40 * 300])
        n_class = np.random.randint(low=2, high=100)
        quantization_type = random.choice(['nbof', 'tnbof'])
        attention_type = random.choice(['spatial', 'temporal', 'spatialsa', 'temporalsa', 'spatiotemporal'])

        learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                     series_length,
                                                     n_class,
                                                     quantization_type=quantization_type,
                                                     attention_type=attention_type,
                                                     iters=1,
                                                     batch_size=4,
                                                     test_mode=True)

        temp_dir = tempfile.TemporaryDirectory()
        learner.save(temp_dir.name, verbose=False)

        new_learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                         series_length,
                                                         n_class,
                                                         quantization_type=quantization_type,
                                                         attention_type=attention_type,
                                                         iters=1,
                                                         batch_size=4,
                                                         test_mode=True)

        new_learner.load(temp_dir.name, verbose=False)
        series = Timeseries(np.random.rand(in_channels, series_length))
        old_pred = learner.infer(series).confidence
        new_pred = new_learner.infer(series).confidence

        self.assertEqual(old_pred, new_pred)
        temp_dir.cleanup()

    def test_download(self):
        in_channels = 1
        series_length = 30 * 300
        n_class = 4
        quantization_type = 'nbof'
        attention_type = 'temporal'
        n_codeword = random.choice([256, 512])
        fold_idx = random.choice([0, 1, 2, 3, 4])

        learner = AttentionNeuralBagOfFeatureLearner(in_channels,
                                                     series_length,
                                                     n_class,
                                                     quantization_type=quantization_type,
                                                     attention_type=attention_type,
                                                     n_codeword=n_codeword)

        temp_dir = tempfile.TemporaryDirectory()
        learner.download(temp_dir.name, fold_idx)
        learner.load(temp_dir.name)
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
