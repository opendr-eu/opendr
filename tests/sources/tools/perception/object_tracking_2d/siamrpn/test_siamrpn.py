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
import gc
import cv2
import shutil
import os
import numpy as np
from opendr.engine.datasets import ExternalDataset
from opendr.engine.target import TrackingAnnotation
from opendr.perception.object_tracking_2d import SiamRPNLearner
from opendr.perception.object_tracking_2d.datasets import OTBTrainDataset


device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'


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


class TestSiamRPNLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST SiamRPN Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_tracking_2d",
                                    "siamrpn", "siamrpn_temp")
        cls.learner = SiamRPNLearner(device=device, temp_path=cls.temp_dir, batch_size=1, n_epochs=1,
                                     lr=1e-4, num_workers=1)
        # Download all required files for testing
        cls.learner.download(cls.temp_dir, mode="pretrained")
        cls.learner.download(os.path.join(cls.temp_dir, "test_data"), mode="test_data")

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for SiamRPN...')
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "siamrpn_opendr"))
        rmdir(os.path.join(cls.temp_dir, "test_data"))
        rmdir(os.path.join(cls.temp_dir))

        del cls.learner
        gc.collect()
        print('Finished cleaning for SiamRPN...')

    def test_fit(self):
        print('Starting training test for SiamRPN...')
        print(os.listdir(os.path.join(self.temp_dir, "test_data")))
        training_dataset = OTBTrainDataset(root=os.path.join(self.temp_dir, "test_data"),
                                           json_path=os.path.join(self.temp_dir, "test_data", "OTBtest.json"))
        m = list(self.learner._model.collect_params().values())[1].data().asnumpy().copy()
        self.learner.fit(dataset=training_dataset, verbose=True)
        n = list(self.learner._model.collect_params().values())[1].data().asnumpy()
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del training_dataset, m, n
        gc.collect()
        print('Finished training test for SiamRPN...')

    def test_eval(self):
        print('Starting evaluation test for SiamRPN...')
        eval_dataset = ExternalDataset(os.path.join(self.temp_dir, "test_data"),
                                       dataset_type="OTBtest")
        self.learner.load(os.path.join(self.temp_dir, "siamrpn_opendr"))
        results_dict = self.learner.eval(eval_dataset)
        self.assertIsNotNone(results_dict['success'],
                             msg="Eval results dictionary not returned.")
        del eval_dataset, results_dict
        gc.collect()
        print('Finished evaluation test for SiamRPN...')

    def test_infer(self):
        print('Starting inference test for SiamRPN...')
        self.learner._model = None
        self.learner.load(os.path.join(self.temp_dir, "siamrpn_opendr"))
        img = cv2.imread(os.path.join(self.temp_dir, "test_data", "Basketball", "img", "0001.jpg"))
        init_box = TrackingAnnotation(left=198, top=214, width=34, height=81, id=0, name=0)
        self.assertIsNotNone(self.learner.infer(img, init_box=init_box),
                             msg="Returned empty TrackingAnnotation.")
        del img
        gc.collect()
        print('Finished inference test for SiamRPN...')

    def test_save_load(self):
        print('Starting save/load test for SiamRPN...')
        self.learner.save(os.path.join(self.temp_dir, "test_model"))
        self.learner._model = None
        self.learner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.learner._model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for SiamRPN...')


if __name__ == "__main__":
    unittest.main()
