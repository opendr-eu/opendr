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
import tempfile
import numpy as np
import random

# OpenDR imports
from opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner import (
    MultilinearCompressiveLearner,
    get_builtin_backbones,
    PRETRAINED_COMPRESSED_SHAPE
)
from opendr.engine.datasets import DatasetIterator
from opendr.engine.data import Image
from opendr.engine.target import Category


class DummyDataset(DatasetIterator):
    def __init__(self, input_shape, compressed_shape, n_class, n_sample=4):
        super(DummyDataset, self).__init__()
        self.input_shape = input_shape
        self.compressed_shape = compressed_shape
        self.n_sample = n_sample
        self.n_class = n_class

    def __len__(self,):
        return self.n_sample

    def __getitem__(self, i):
        x = np.random.rand(*self.input_shape)
        y = np.random.randint(low=0, high=self.n_class)
        return Image(x), Category(y)


class DummyBackbone(torch.nn.Module):
    def __init__(self, input_shape, n_class):
        super(DummyBackbone, self).__init__()

        n_element = input_shape[0] * input_shape[1] * input_shape[2]
        self.linear = torch.nn.Linear(n_element, n_class)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


def get_random_learner():
    backbone = random.choice(get_builtin_backbones())
    if backbone.startswith('cifar'):
        input_shape = (32, 32, 3)
        compressed_shape = (6, 6, 1)
    else:
        input_shape = (224, 224, 3)
        compressed_shape = (6, 6, 2)
    n_class = np.random.randint(low=2, high=100)
    pretrained_backbone = random.choice(['', 'without_classifier'])
    init_backbone = random.choice([True, False])
    learner = MultilinearCompressiveLearner(input_shape=input_shape,
                                            compressed_shape=compressed_shape,
                                            backbone=backbone,
                                            n_class=n_class,
                                            pretrained_backbone=pretrained_backbone,
                                            init_backbone=init_backbone,
                                            n_init_iters=1,
                                            iters=1,
                                            batch_size=2,
                                            test_mode=True)

    return learner, input_shape, compressed_shape, n_class, pretrained_backbone, init_backbone


class TestMultilinearCompressiveLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Multilinear Compressive Learner\n"
              "**********************************")
        pass

    @classmethod
    def tearDownClass(cls):
        return

    def test_fit_with_custom_backbone(self):
        input_shape = (32, 32, 3)
        compressed_shape = (6, 6, 2)
        n_class = np.random.randint(low=2, high=100)

        train_set = DummyDataset(input_shape, compressed_shape, n_class)
        val_set = DummyDataset(input_shape, compressed_shape, n_class)
        test_set = DummyDataset(input_shape, compressed_shape, n_class)

        backbone = DummyBackbone(input_shape, n_class)

        learner = MultilinearCompressiveLearner(input_shape=input_shape,
                                                compressed_shape=compressed_shape,
                                                backbone=backbone,
                                                n_class=n_class,
                                                pretrained_backbone='',
                                                init_backbone=True,
                                                n_init_iters=1,
                                                iters=1,
                                                batch_size=4,
                                                test_mode=True)

        old_weight = list(learner.model.parameters())[0].clone()
        learner.fit(train_set, val_set, test_set, silent=True, verbose=False)
        new_weight = list(learner.model.parameters())[0].clone()

        self.assertFalse(torch.equal(old_weight, new_weight),
                         msg="Model parameters did not change after running fit.")

    def test_fit_with_builtin_backbone(self):
        learner, input_shape, compressed_shape, n_class, pretrained_backbone, init_backbone = get_random_learner()

        train_set = DummyDataset(input_shape, compressed_shape, n_class)
        val_set = DummyDataset(input_shape, compressed_shape, n_class)
        test_set = DummyDataset(input_shape, compressed_shape, n_class)

        old_weight = list(learner.model.parameters())[0].clone()
        learner.fit(train_set, val_set, test_set, silent=True, verbose=False)
        new_weight = list(learner.model.parameters())[0].clone()

        self.assertFalse(torch.equal(old_weight, new_weight),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        learner, input_shape, compressed_shape, n_class, pretrained_backbone, init_backbone = get_random_learner()

        dataset = DummyDataset(input_shape, compressed_shape, n_class)
        performance = learner.eval(dataset, silent=True, verbose=False)

        self.assertTrue('cross_entropy' in performance.keys())
        self.assertTrue('acc' in performance.keys())

    def test_infer(self):
        learner, input_shape, compressed_shape, n_class, pretrained_backbone, init_backbone = get_random_learner()
        img = Image(np.random.rand(*input_shape))
        pred = learner.infer(img)
        self.assertTrue(isinstance(pred, Category))
        self.assertTrue(pred.data < learner.n_class,
                        msg="Predicted class label must be less than the number of class")
        self.assertTrue(pred.confidence <= 1,
                        msg="Confidence of prediction must be less or equal than 1")

    def test_infer_from_compressed_measurement(self):
        learner, input_shape, compressed_shape, n_class, pretrained_backbone, init_backbone = get_random_learner()
        img = Image(np.random.rand(*compressed_shape))
        pred = learner.infer_from_compressed_measurement(img)
        self.assertTrue(isinstance(pred, Category))
        self.assertTrue(pred.data < learner.n_class,
                        msg="Predicted class label must be less than the number of class")
        self.assertTrue(pred.confidence <= 1,
                        msg="Confidence of prediction must be less or equal than 1")

    def test_save_load(self):
        learner, input_shape, compressed_shape, n_class, _, _ = get_random_learner()
        temp_dir = tempfile.TemporaryDirectory()
        learner.save(temp_dir.name, verbose=False)
        new_learner = MultilinearCompressiveLearner(input_shape=input_shape,
                                                    compressed_shape=compressed_shape,
                                                    backbone=learner.backbone_classifier,
                                                    n_class=n_class,
                                                    pretrained_backbone='',
                                                    init_backbone=True,
                                                    n_init_iters=1,
                                                    iters=1,
                                                    batch_size=4,
                                                    test_mode=True)

        new_learner.load(temp_dir.name, verbose=False)
        img = Image(np.random.rand(*input_shape))
        old_pred = learner.infer(img).confidence
        new_pred = new_learner.infer(img).confidence

        self.assertEqual(old_pred, new_pred)
        temp_dir.cleanup()

    def test_download(self):
        backbone = 'cifar_allcnn'
        input_shape = (32, 32, 3)
        compressed_shape = random.choice(PRETRAINED_COMPRESSED_SHAPE)

        n_class = random.choice([10, 100])
        learner = MultilinearCompressiveLearner(input_shape=input_shape,
                                                compressed_shape=compressed_shape,
                                                backbone=backbone,
                                                n_class=n_class)

        temp_dir = tempfile.TemporaryDirectory()
        learner.download(temp_dir.name)

        metadata_file = os.path.join(temp_dir.name, 'metadata.json')
        weights_file = os.path.join(temp_dir.name, 'model_weights.pt')
        self.assertTrue(os.path.exists(metadata_file),
                        msg='metadata file was not downloaded')
        self.assertTrue(os.path.exists(weights_file),
                        msg='model weights file was not downloaded')

        learner.load(temp_dir.name)
        temp_dir.cleanup()

    def test_get_sensing_parameters(self):
        learner, input_shape, compressed_shape, n_class, _, _ = get_random_learner()
        params = learner.get_sensing_parameters()
        self.assertTrue(len(params) in [2, 3],
                        msg='sensing parameters should be a list of 2 or 3 elements')

        for idx, param in enumerate(params):
            self.assertTrue(isinstance(param, np.ndarray),
                            msg='each sensing parameter must be an instance of numpy.ndarray')
            correct_shape = (input_shape[idx], compressed_shape[idx])
            self.assertTrue(param.shape == correct_shape,
                            msg='the {}-th parameter should have shape: {}'.format(idx, correct_shape))


if __name__ == "__main__":
    unittest.main()
