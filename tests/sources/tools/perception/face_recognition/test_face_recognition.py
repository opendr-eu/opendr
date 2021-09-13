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

import numpy as np
import os
import shutil
import unittest
from opendr.perception.face_recognition.face_recognition_learner import FaceRecognitionLearner
from opendr.engine.datasets import ExternalDataset


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


class TestFaceRecognitionLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = './face_recognition_temp'
        cls.recognizer = FaceRecognitionLearner(backbone='mobilefacenet', mode='backbone_only', device="cpu",
                                                temp_path=cls.temp_dir, batch_size=10, checkpoint_after_iter=0)
        # Download all required files for testing
        cls.recognizer.download(cls.temp_dir, mode='pretrained')
        cls.recognizer.download(cls.temp_dir, mode="test_data")

    @classmethod
    def tearDownClass(cls):
        rmdir(os.path.join(cls.temp_dir, "test_data"))
        rmdir(cls.temp_dir)

    def test_fit(self):
        recognizer = FaceRecognitionLearner(backbone='mobilefacenet', mode='full', device="cpu",
                                            temp_path=self.temp_dir, iters=2,
                                            batch_size=2, checkpoint_after_iter=0)
        dataset_path = os.path.join(self.temp_dir, 'test_data/images')
        train = ExternalDataset(path=dataset_path, dataset_type='imagefolder')
        results = recognizer.fit(dataset=train, silent=False, verbose=True)
        self.assertNotEqual(len(results), 0)

    def test_align(self):
        imgs = os.path.join(self.temp_dir, 'test_data/images')
        self.recognizer.load(self.temp_dir)
        self.recognizer.align(imgs, os.path.join(self.temp_dir, 'aligned'))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'aligned')))
        # Cleanup
        rmdir(os.path.join(self.temp_dir, 'aligned'))

    def test_fit_reference(self):
        imgs = os.path.join(self.temp_dir, 'test_data/images')
        save_path = os.path.join(self.temp_dir, 'reference')
        self.recognizer.load(self.temp_dir)
        self.recognizer.fit_reference(imgs, save_path)
        self.assertTrue(os.path.exists(os.path.join(save_path, 'reference.pkl')))
        # Cleanup
        rmfile(os.path.join(self.temp_dir, 'reference', 'reference.pkl'))
        rmdir(os.path.join(self.temp_dir, 'reference'))

    def test_infer(self):
        imgs = os.path.join(self.temp_dir, 'test_data/images')
        save_path = os.path.join(self.temp_dir, 'reference')
        self.recognizer.load(self.temp_dir)
        self.recognizer.fit_reference(imgs, save_path)
        img = np.random.random((112, 112, 3))
        result = self.recognizer.infer(img)
        self.assertIsNotNone(result)
        # Cleanup
        rmfile(os.path.join(self.temp_dir, 'reference', 'reference.pkl'))
        rmdir(os.path.join(self.temp_dir, 'reference'))

    def test_eval(self):
        self.recognizer.load(self.temp_dir)
        dataset_path = os.path.join(self.temp_dir, 'test_data/images')
        eval_dataset = ExternalDataset(path=dataset_path, dataset_type='imagefolder')
        results = self.recognizer.eval(eval_dataset, num_pairs=10000)
        self.assertNotEqual(len(results), 0)

    def test_save_load(self):
        save_path = os.path.join(self.temp_dir, 'saved')
        self.recognizer.backbone_model = None
        self.recognizer.load(self.temp_dir)
        self.assertIsNotNone(self.recognizer.backbone_model, "model is None after loading pth model.")
        self.recognizer.save(save_path)
        self.assertTrue(os.path.exists(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.pth')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.json')))
        # Cleanup
        rmfile(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.pth'))
        rmfile(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.json'))
        rmdir(os.path.join(self.temp_dir, 'saved'))

    def test_optimize(self):
        self.recognizer.load(self.temp_dir)
        self.recognizer.optimize()
        self.assertIsNotNone(self.recognizer.ort_backbone_session)
        self.recognizer.ort_backbone_session = None
        # Cleanup
        rmfile(os.path.join(self.temp_dir, 'onnx_' + self.recognizer.backbone + '_backbone_model.onnx'))

    def test_download(self):
        download_path = os.path.join(self.temp_dir, 'downloaded')
        check_path = os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.pth')
        check_path_json = os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.json')
        self.recognizer.download(download_path, mode="pretrained")
        self.assertTrue(os.path.exists(check_path))
        self.assertTrue(os.path.exists(check_path_json))
        # Cleanup
        rmfile(os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.pth'))
        rmfile(os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.json'))
        rmdir(os.path.join(self.temp_dir, 'downloaded'))


if __name__ == '__main__':
    unittest.main()
