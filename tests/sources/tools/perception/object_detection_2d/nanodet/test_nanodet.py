# Copyright 2020-2022 OpenDR European Project
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
import gc
import shutil
import os
import numpy as np
from opendr.perception.object_detection_2d import NanodetLearner
from opendr.engine.datasets import ExternalDataset
import torch

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

_DEFAULT_MODEL = "plus-m_416"

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

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "nanodet", "nanodet_temp")
        cls.detector = NanodetLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=cls.temp_dir, batch_size=1,
                                      iters=1, checkpoint_after_iter=0, lr=1e-4)
        # Download all required files for testing
        cls.detector.download(path=cls.temp_dir, model="pretrained")
        cls.detector.download(path=cls.temp_dir, mode="images")
        cls.detector.download(path=cls.temp_dir, mode="test_data")

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for Nanodet...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "000000000036.jpg"))
        rmdir(os.path.join(cls.temp_dir, "test_data"))
        rmdir(os.path.join(cls.temp_dir, "nanodet-plus-m_416"))
        rmdir(os.path.join(cls.temp_dir))

        del cls.detector
        gc.collect()
        print('Finished cleaning for Nanodet...')

    def test_fit(self):
        print('Starting training test for Nanodet...')
        training_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "test_data"), dataset_type="voc")
        m = list(self.detector._model.collect_params().values())[2].data().asnumpy().copy()
        self.detector.fit(dataset=training_dataset, verbose=False)
        n = list(self.detector._model.collect_params().values())[2].data().asnumpy()
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del training_dataset, m, n
        gc.collect()
        print('Finished training test for Nanodet...')

    def test_eval(self):
        print('Starting evaluation test for Nanodet...')
        eval_dataset = ExternalDataset(path=os.path.join(self.temp_dir, "test_data"), dataset_type="voc")
        self.detector.load(path=os.path.join(self.temp_dir, f"nanodet_{_DEFAULT_MODEL}", f"nanodet_{_DEFAULT_MODEL}.ckpt"), verbose=False)
        results_dict = self.detector.eval(dataset=eval_dataset, verbose=False)
        self.assertIsNotNone(results_dict['map'],
                             msg="Eval results dictionary not returned.")
        del eval_dataset, results_dict
        gc.collect()
        print('Finished evaluation test for Nanodet...')

    def test_infer(self):
        print('Starting inference test for Nanodet...')
        self.detector.load(os.path.join(self.temp_dir, f"nanodet_{_DEFAULT_MODEL}", f"nanodet_{_DEFAULT_MODEL}.ckpt"))
        self.assertIsNotNone(self.detector.infer(input=os.path.join(self.temp_dir, "000000000036.jpg"), show=False),
                             msg="Returned empty BoundingBoxList.")
        gc.collect()
        print('Finished inference test for Nanodet...')

    def test_save_load(self):
        print('Starting save/load test for Nanodet...')
        self.detector.save(path=os.path.join(self.temp_dir, "test_model"), verbose=False)
        starting_param_1 = list(self.detector.model.parameters())[0].clone()
        self.detector.model = None
        detector2 = NanodetLearner(model_to_use=_DEFAULT_MODEL, device=device, temp_path=self.temp_dir, batch_size=1,
                                   iters=1, checkpoint_after_iter=0, lr=1e-4)
        detector2.load(path=os.path.join(self.temp_dir, "test_model", "save.pth"), verbose=False)
        new_param = list(detector2.model.parameters())[0].clone()
        self.assertTrue(torch.equal(starting_param_1, new_param))

        # Cleanup
        rmfile(os.path.join(self.temp_dir, "test_model", "save.pth"))
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for Nanodet...')


if __name__ == "__main__":
    unittest.main()
