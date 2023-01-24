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
from opendr.perception.object_detection_2d import NanodetLearner
from opendr.engine.datasets import ExternalDataset

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

_DEFAULT_MODEL = "m"


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


class TestNanodetLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Nanodet Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "nanodet_temp")
        cls.detector = NanodetLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=cls.temp_dir, batch_size=1,
                                      iters=1, checkpoint_after_iter=2, lr=1e-4)
        # Download all required files for testing
        cls.detector.download(path=cls.temp_dir, mode="pretrained", verbose=False)
        cls.detector.download(path=cls.temp_dir, mode="images", verbose=False)
        cls.detector.download(path=cls.temp_dir, mode="test_data", verbose=False)

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for Nanodet...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "000000000036.jpg"))
        rmdir(os.path.join(cls.temp_dir, "test_data"))
        rmdir(os.path.join(cls.temp_dir, "nanodet_{}".format(_DEFAULT_MODEL)))
        rmdir(os.path.join(cls.temp_dir))

        del cls.detector
        gc.collect()
        print('Finished cleaning for Nanodet...')

    def test_fit(self):
        print('Starting training test for Nanodet...')
        training_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "test_data"), dataset_type="voc")
        m = list(self.detector._model.parameters())[0].clone().detach().clone().to(device)
        self.detector.fit(dataset=training_dataset, verbose=False)
        n = list(self.detector._model.parameters())[0].clone().detach().clone().to(device)
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del training_dataset, m, n
        gc.collect()

        rmfile(os.path.join(self.temp_dir, "checkpoints", "model_iter_0.ckpt"))
        rmfile(os.path.join(self.temp_dir, "checkpoints", "epoch=0-step=0.ckpt"))
        rmdir(os.path.join(self.temp_dir, "checkpoints"))

        print('Finished training test for Nanodet...')

    def test_eval(self):
        print('Starting evaluation test for Nanodet...')
        eval_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "test_data"), dataset_type="voc")
        self.detector.load(path=os.path.join(self.temp_dir, "nanodet_{}".format(_DEFAULT_MODEL)), verbose=False)
        results_dict = self.detector.eval(dataset=eval_dataset, verbose=False)
        self.assertNotEqual(len(results_dict), 0,
                            msg="Eval results dictionary list is empty.")
        del eval_dataset, results_dict
        gc.collect()

        rmfile(os.path.join(self.temp_dir, "results.json"))
        rmfile(os.path.join(self.temp_dir, "eval_results.txt"))
        print('Finished evaluation test for Nanodet...')

    def test_infer(self):
        print('Starting inference test for Nanodet...')
        self.detector.load(os.path.join(self.temp_dir, "nanodet_{}".format(_DEFAULT_MODEL)), verbose=False)
        img = cv2.imread(os.path.join(self.temp_dir, "000000000036.jpg"))
        self.assertIsNotNone(self.detector.infer(input=img),
                             msg="Returned empty BoundingBoxList.")
        gc.collect()
        print('Finished inference test for Nanodet...')

    def test_save_load(self):
        print('Starting save/load test for Nanodet...')
        self.detector.ort_session = None
        self.detector.jit_model = None
        self.detector.save(path=os.path.join(self.temp_dir, "test_model"), verbose=False)
        starting_param_1 = list(self.detector._model.parameters())[0].detach().clone().to(device)
        self.detector.model = None
        detector2 = NanodetLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=self.temp_dir, batch_size=1,
                                   iters=1, checkpoint_after_iter=1, lr=1e-4)
        detector2.load(path=os.path.join(self.temp_dir, "test_model"), verbose=False)
        new_param = list(detector2._model.parameters())[0].detach().clone().to(device)
        self.assertTrue(starting_param_1.allclose(new_param))

        del starting_param_1, new_param
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "test_model", "nanodet_{}.json".format(_DEFAULT_MODEL)))
        rmfile(os.path.join(self.temp_dir, "test_model", "nanodet_{}.pth".format(_DEFAULT_MODEL)))
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for Nanodet...')

    def test_optimize(self):
        # Tracing will issue TracerWarnings, but these can be ignored safely
        # because we use this function to create tensors out of constant
        # variables that are the same every time we call this function.
        warnings.simplefilter("ignore",  TracerWarning)
        warnings.simplefilter("ignore",  RuntimeWarning)

        self.detector.ort_session = None
        self.detector.jit_model = None

        self.detector.optimize(os.path.join(self.temp_dir, "onnx"), verbose=False, optimization="onnx")
        self.assertIsNotNone(self.detector.ort_session)

        self.detector.optimize(os.path.join(self.temp_dir, "jit"), verbose=False, optimization="jit")
        self.assertIsNotNone(self.detector.jit_model)

        # Cleanup
        rmfile(os.path.join(self.temp_dir, "onnx", "nanodet_{}.onnx".format(_DEFAULT_MODEL)))
        rmfile(os.path.join(self.temp_dir, "onnx", "nanodet_{}.json".format(_DEFAULT_MODEL)))
        rmfile(os.path.join(self.temp_dir, "jit", "nanodet_{}.pth".format(_DEFAULT_MODEL)))
        rmfile(os.path.join(self.temp_dir, "jit", "nanodet_{}.json".format(_DEFAULT_MODEL)))
        rmdir(os.path.join(self.temp_dir, "onnx"))
        rmdir(os.path.join(self.temp_dir, "jit"))

        warnings.simplefilter("default",  TracerWarning)
        warnings.simplefilter("default",  RuntimeWarning)


if __name__ == "__main__":
    unittest.main()
