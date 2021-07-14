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
import gc
import cv2
import shutil
import os
import numpy as np
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.datasets.wider_person import WiderPersonDataset


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


class TestSSDLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST SingleShotDetector Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "ssd", "ssd_temp")
        cls.detector = SingleShotDetectorLearner(device="cpu", temp_path=cls.temp_dir, batch_size=1, epochs=1,
                                                 checkpoint_after_iter=0, lr=1e-4, num_workers=0)
        # Download all required files for testing
        cls.detector.download(mode="pretrained")
        cls.detector.download(mode="images")
        cls.detector.download(mode="test_data")

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for SSD...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "people.jpg"))
        rmdir(os.path.join(cls.temp_dir, "ssd_default_person"))
        rmdir(os.path.join(cls.temp_dir, "test_data"))
        rmdir(os.path.join(cls.temp_dir))

        del cls.detector
        gc.collect()
        print('Finished cleaning for SSD...')

    def test_fit(self):
        print('Starting training test for SSD...')
        training_dataset = WiderPersonDataset(root=os.path.join(self.temp_dir, "test_data"), splits=['train'])
        m = list(self.detector._model.collect_params().values())[2].data().asnumpy().copy()
        self.detector.fit(dataset=training_dataset, silent=True)
        n = list(self.detector._model.collect_params().values())[2].data().asnumpy()
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del training_dataset, m, n
        gc.collect()
        print('Finished training test for SSD...')

    def test_eval(self):
        print('Starting evaluation test for SSD...')
        eval_dataset = WiderPersonDataset(root=os.path.join(self.temp_dir, "test_data"), splits=['train'])
        self.detector.load(os.path.join(self.temp_dir, "ssd_default_person"))
        results_dict = self.detector.eval(eval_dataset)
        self.assertIsNotNone(results_dict['map'],
                             msg="Eval results dictionary not returned.")
        del eval_dataset, results_dict
        gc.collect()
        print('Finished evaluation test for SSD...')

    def test_infer(self):
        print('Starting inference test for SSD...')
        self.detector.load(os.path.join(self.temp_dir, "ssd_default_person"))
        img = cv2.imread(os.path.join(self.temp_dir, "people.jpg"))
        self.assertIsNotNone(self.detector.infer(img),
                             msg="Returned empty BoundingBoxList.")
        del img
        gc.collect()
        print('Finished inference test for SSD...')

    def test_save_load(self):
        print('Starting save/load test for SSD...')
        self.detector.save(os.path.join(self.temp_dir, "test_model"))
        self.detector._model = None
        self.detector.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.detector._model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for SSD...')


if __name__ == "__main__":
    unittest.main()
