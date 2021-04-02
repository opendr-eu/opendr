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
import shutil
import os
import torch
import numpy as np
from perception.skeleton_based_action_recognition.stgcn_learner import STGCNLearner
from engine.datasets import ExternalDataset


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


# PATH_ = os.path.join(".", "tests", "sources", "tools", "perception", "skeleton_based_action_recognition")
# LOG_PATH_ = os.path.join(".", "tests", "sources", "tools", "perception", "skeleton_based_action_recognition", "logs")
PATH_ = './tests/sources/tools/perception/skeleton_based_action_recognition'
LOG_PATH_ = ''


class TestSkeletonBasedActionRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.stbln_action_classifier = STGCNLearner(device="cpu", temp_path=cls.temp_dir, batch_size=1, epochs=1,
                                                   checkpoint_after_iter=1, val_batch_size=1,
                                                   dataset_name='nturgbd_cv', experiment_name='stbln_nturgbd',
                                                   method_name='stbln', stbln_symmetric=False)
        # Download all required files for testing
        '''cls.Pretrained_MODEL_PATH = cls.stgcn_action_classifier.download(
            mode="pretrained", path=os.path.join(cls.temp_dir, "pretrained_models"))
        cls.Train_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Val_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Test_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="test_data", path=os.path.join(cls.temp_dir, "data"))'''

        cls.Pretrained_MODEL_PATH = os.path.join(cls.temp_dir, "pretrained_models")
        cls.Train_DATASET_PATH = os.path.join(cls.temp_dir, "data", 'nturgbd_cv')
        cls.Val_DATASET_PATH = os.path.join(cls.temp_dir, "data", 'nturgbd_cv')
        cls.Test_DATASET_PATH = os.path.join(cls.temp_dir, "data", 'nturgbd_cv', 'val_joints.npy')

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "pretrained_models"))
        rmdir(os.path.join(cls.temp_dir, "data"))
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.init_model()
        m = list(self.stbln_action_classifier.model.parameters())[0].clone()
        self.stbln_action_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset, silent=True,
                                         train_data_filename='train_joints.npy',
                                         train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
                                         val_labels_filename="val_labels.pkl")
        self.assertFalse(torch.equal(m, list(self.stbln_action_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'PretrainedModel'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.stbln_action_classifier.load(model_saved_path, model_name)
        score_dict = self.stbln_action_classifier.eval(validation_dataset, val_data_filename='val_joints.npy',
                                                       val_labels_filename='val_labels.pkl')
        self.assertNotEqual(len(score_dict), 0,
                            msg="Eval results dictionary contains empty list.")

    def test_infer(self):
        test_data = np.load(self.Test_DATASET_PATH)
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'PretrainedModel'
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.load(model_saved_path, model_name)
        model_output = self.stbln_action_classifier.infer(test_data)
        self.assertIsNotNone(model_output, msg="The model output is None")

    def test_save_load(self):
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.ort_session = None
        self.stbln_action_classifier.init_model()
        self.stbln_action_classifier.save(path=os.path.join(self.temp_dir, "test_save_load"), model_name='testModel')
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.load(path=os.path.join(self.temp_dir, "test_save_load"), model_name='testModel')
        self.assertIsNotNone(self.stbln_action_classifier.model, "model is None after loading pt model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_save_load"))

    def test_save_load_onnx(self):
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.ort_session = None
        self.stbln_action_classifier.init_model()
        self.stbln_action_classifier.optimize()
        self.stbln_action_classifier.save(path=os.path.join(self.temp_dir, "test_save_load"),
                                          model_name='onnx_model_temp')
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.load(path=os.path.join(self.temp_dir, "test_save_load"),
                                          model_name='onnx_model_temp')
        self.assertIsNotNone(self.stbln_action_classifier.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))
        rmdir(os.path.join(self.temp_dir, "testOnnxModel"))
        self.stbln_action_classifier.ort_session = None

    def test_optimize(self):
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'PretrainedModel'
        self.stbln_action_classifier.model = None
        self.stbln_action_classifier.ort_session = None
        self.stbln_action_classifier.load(model_saved_path, model_name)
        self.stbln_action_classifier.optimize()
        self.assertIsNotNone(self.stbln_action_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.stbln_action_classifier.ort_session = None
        rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))


if __name__ == "__main__":
    unittest.main()