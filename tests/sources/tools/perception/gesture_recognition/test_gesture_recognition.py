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

import cv2
import unittest
import gc
import shutil
import os
import warnings
from torch.jit import TracerWarning
import numpy as np
from opendr.perception.gesture_recognition.gesture_recognition_learner import GestureRecognitionLearner
from opendr.engine.datasets import ExternalDataset
import json
import numpy as np
import time
device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

_DEFAULT_MODEL = "plus_m_1.5x_416"


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


def make_dummy_dataset(tmp_dir):
    os.makedirs(tmp_dir)
    idi = 0
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(tmp_dir, split))
        annotations = []
        categories = []
        images = []
        classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted', 'three', 'two_up', 'two_up_inverted', 'three2', 'peace_inverted', 'no_gesture']
        for i, name in enumerate(classes):
            categories.append({'supercategory': 'none', 'name': name, 'id': i})
            dummy_image = np.zeros((416, 416, 3))
            cv2.imwrite(os.path.join(tmp_dir, split, 'image_{}_{}_{}.jpg'.format(name, split, 0)), dummy_image)
            dummy_image2 = np.zeros((416, 416, 3))
            cv2.imwrite(os.path.join(tmp_dir, split, 'image_{}_{}_{}.jpg'.format(name, split, 1)), dummy_image2)
            images.append({'file_name': 'image_{}_{}_{}.jpg'.format(name, split, 0), 'height': 416, 'width': 416, 'id': idi})
            images.append({'file_name': 'image_{}_{}_{}.jpg'.format(name, split, 1), 'height': 416, 'width': 416, 'id': idi+1})
            annotations.append({'id': idi, 'bbox': [233.8257552, 238.43682560000002, 118.39368, 145.3367104], 'segmentation': [[233.8257552, 238.43682560000002, 352.2194352, 238.43682560000002, 352.2194352, 383.77353600000004, 233.8257552, 383.77353600000004]], 'image_id': idi, 'category_id': i, 'iscrowd': 0, 'area': 17206.94798335027})
            annotations.append({'id': idi+1, 'bbox': [233.8257552, 238.43682560000002, 118.39368, 145.3367104], 'segmentation': [[233.8257552, 238.43682560000002, 352.2194352, 238.43682560000002, 352.2194352, 383.77353600000004, 233.8257552, 383.77353600000004]], 'image_id': idi+1, 'category_id': i, 'iscrowd': 0, 'area': 17206.94798335027})
            idi += 2
        temp = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(os.path.join(tmp_dir, '{}.json'.format(split)), 'w') as f:
            json.dump(temp, f)


class TestGestureRecognitionLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Nanodet Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", 'temp_gesture_'+str(time.time()))
        cls.model = GestureRecognitionLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=cls.temp_dir, batch_size=1, iters=1, checkpoint_after_iter=2, lr=1e-4)
        make_dummy_dataset(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directory for Gesture recognition...')
        # Clean up downloaded files
        rmdir(cls.temp_dir)

        del cls.model
        gc.collect()
        print('Finished cleaning for Gesture recognition...')

    def test_fit(self):
        print('Starting training test for Gesture recognition...')
        dataset = ExternalDataset(self.temp_dir, 'coco')
        val_dataset = ExternalDataset(self.temp_dir, 'coco')
        m = list(self.model._model.parameters())[0].clone().detach().clone().to(device)
        self.model.fit(dataset=dataset, val_dataset=val_dataset, verbose=False)
        n = list(self.model._model.parameters())[0].clone().detach().clone().to(device)
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del dataset, m, n
        gc.collect()

        rmfile(os.path.join(self.temp_dir, "checkpoints", "model_iter_0.ckpt"))
        rmdir(os.path.join(self.temp_dir, "checkpoints"))

        print('Finished training test for Gesture recognition...')

    def test_eval(self):
        print('Starting evaluation test for Gesture recognition...')

        test_dataset = ExternalDataset(self.temp_dir, 'coco')
        results_dict = self.model.eval(dataset=test_dataset, verbose=False)
        self.assertNotEqual(len(results_dict), 0,
                            msg="Eval results dictionary list is empty.")

        del test_dataset, results_dict
        gc.collect()

        rmfile(os.path.join(self.temp_dir, "eval_results.txt"))
        print('Finished evaluation test for Nanodet...')

    def test_save_load(self):
        print('Starting save/load test for Nanodet...')
        self.model.save(path=os.path.join(self.temp_dir, "test_model"), verbose=False)
        starting_param_1 = list(self.model._model.parameters())[0].detach().clone().to(device)
        self.model.model = None
        learner2 = GestureRecognitionLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=self.temp_dir, batch_size=1,
        iters=1, checkpoint_after_iter=1, lr=1e-4)
        learner2.load(path=os.path.join(self.temp_dir, "test_model"), verbose=False)
        new_param = list(learner2._model.parameters())[0].detach().clone().to(device)
        self.assertTrue(starting_param_1.allclose(new_param))

        del starting_param_1, new_param
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "test_model", "nanodet_{}.json".format(_DEFAULT_MODEL)))
        rmfile(os.path.join(self.temp_dir, "test_model", "nanodet_{}.pth".format(_DEFAULT_MODEL)))
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for Gesture learner...')


if __name__ == "__main__":
    unittest.main()
