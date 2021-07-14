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
from opendr.perception.object_detection_2d.retinaface.retinaface_learner import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.wider_face import WiderFaceDataset


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


class TestRetinaFaceLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST RetinaFace Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "retinaface", "retinaface_temp")
        cls.detector = RetinaFaceLearner(device="cpu", temp_path=cls.temp_dir, batch_size=1, epochs=1,
                                         checkpoint_after_iter=0, lr=1e-4)
        # Download all required files for testing
        cls.detector.download(mode="pretrained")
        cls.detector.download(mode="images")
        cls.detector.download(mode="test_data")

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for RetinaFace...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "cov4.jpg"))
        rmdir(os.path.join(cls.temp_dir, "retinaface_resnet"))
        rmdir(os.path.join(cls.temp_dir))

        del cls.detector
        gc.collect()
        print('Finished cleaning for RetinaFace...')

    def test_fit(self):
        print('Starting training test for RetinaFace...')
        training_dataset = WiderFaceDataset(root=self.temp_dir, splits=['train'])
        m = list(self.detector._model.get_params()[0].values())[0].asnumpy().copy()
        self.detector.fit(dataset=training_dataset, silent=True)
        n = list(self.detector._model.get_params()[0].values())[0].asnumpy()
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del training_dataset, m, n
        gc.collect()
        print('Finished training test for RetinaFace...')

    def test_eval(self):
        print('Starting evaluation test for RetinaFace...')
        eval_dataset = WiderFaceDataset(root=self.temp_dir, splits=['train'])
        self.detector.load(os.path.join(self.temp_dir, "retinaface_resnet"))
        results_dict = self.detector.eval(eval_dataset, flip=False, pyramid=False)
        self.assertIsNotNone(results_dict['recall'],
                             msg="Eval results dictionary not returned.")
        del eval_dataset, results_dict
        gc.collect()
        print('Finished evaluation test for RetinaFace...')

    def test_infer(self):
        print('Starting inference test for RetinaFace...')
        self.detector.load(os.path.join(self.temp_dir, "retinaface_resnet"))
        img = cv2.imread(os.path.join(self.temp_dir, "cov4.jpg"))
        self.assertIsNotNone(self.detector.infer(img),
                             msg="Returned empty BoundinBoxList.")
        del img
        gc.collect()
        print('Finished inference test for RetinaFace...')

    def test_save_load(self):
        print('Starting save/load test for RetinaFace...')
        self.detector.save(os.path.join(self.temp_dir, "test_model"))
        self.detector._model = None
        self.detector.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.detector._model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for RetinaFace...')


if __name__ == "__main__":
    unittest.main()
