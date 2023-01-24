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
import shutil
import torch
import numpy as np
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner
from opendr.engine.datasets import ExternalDataset
import os

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


PATH_ = './tests/sources/tools/perception/skeleton_based_action_recognition/skeleton_based_action_recognition_temp'
LOG_PATH_ = ''


class TestSkeletonBasedActionRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST Skeleton Based Action Recognition Learner with STGCN model\n*"
            "*********************************")
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.stgcn_action_classifier = SpatioTemporalGCNLearner(device=device, temp_path=cls.temp_dir,
                                                               batch_size=2, epochs=1,
                                                               checkpoint_after_iter=1, val_batch_size=2,
                                                               dataset_name='nturgbd_cv',
                                                               num_class=60, num_point=25, num_person=2, in_channels=3,
                                                               graph_type='ntu',
                                                               experiment_name='stgcn_nturgbd_cv_joint',
                                                               method_name='stgcn')
        cls.experiment_name = 'stgcn_nturgbd_cv_joint'
        # Download all required files for testing
        cls.Pretrained_MODEL_PATH_J = cls.stgcn_action_classifier.download(
            path=os.path.join(cls.temp_dir, "pretrained_models", 'stgcn'), method_name="stgcn", mode="pretrained",
            file_name='stgcn_nturgbd_cv_joint-49-29400')
        cls.Train_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Val_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Test_DATASET_PATH = cls.stgcn_action_classifier.download(
            mode="test_data", path=os.path.join(cls.temp_dir, "data"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "data"))
        rmdir(os.path.join(cls.temp_dir, "pretrained_models"))
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        print(
            "\n\n**********************************\nTest STGCN fit function \n*"
            "*********************************")
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.init_model()
        m = list(self.stgcn_action_classifier.model.parameters())[0].clone()
        self.stgcn_action_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset, silent=True,
                                         verbose=False, train_data_filename='train_joints.npy',
                                         train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
                                         val_labels_filename="val_labels.pkl",
                                         skeleton_data_type='joint')
        self.assertFalse(torch.equal(m, list(self.stgcn_action_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        print(
            "\n\n**********************************\nTest STGCN eval function \n*"
            "*********************************")
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'stgcn_nturgbd_cv_joint-49-29400'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.stgcn_action_classifier.load(model_saved_path, model_name)
        eval_results = self.stgcn_action_classifier.eval(validation_dataset, verbose=False,
                                                         val_data_filename='val_joints.npy',
                                                         val_labels_filename='val_labels.pkl',
                                                         skeleton_data_type='joint')
        self.assertNotEqual(len(eval_results["score"]), 0,
                            msg="Eval results contains empty list.")

    def test_infer(self):
        print(
            "\n\n**********************************\nTest STGCN infer function \n*"
            "*********************************")
        test_data = np.load(self.Test_DATASET_PATH)[0:1]
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'stgcn_nturgbd_cv_joint-49-29400'
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.load(model_saved_path, model_name)
        category = self.stgcn_action_classifier.infer(test_data)
        self.assertIsNotNone(category.confidence, msg="The predicted confidence score is None")

    def test_save_load(self):
        print(
            "\n\n**********************************\nTest STGCN save_load function \n*"
            "*********************************")
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.ort_session = None
        self.stgcn_action_classifier.init_model()
        self.stgcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='test_stgcn')
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='test_stgcn')
        self.assertIsNotNone(self.stgcn_action_classifier.model, "model is None after loading pt model.")
        # Cleanup

    def test_save_load_onnx(self):
        print(
            "\n\n**********************************\nTest STGCN saveload ONNX function \n*"
            "*********************************")
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.ort_session = None
        self.stgcn_action_classifier.init_model()
        self.stgcn_action_classifier.optimize()
        self.stgcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='onnx_model_temp')
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='onnx_model_temp')
        self.assertIsNotNone(self.stgcn_action_classifier.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        self.stgcn_action_classifier.ort_session = None

    def test_optimize(self):
        print(
            "\n\n**********************************\nTest STGCN optimize function \n*"
            "*********************************")
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'stgcn_nturgbd_cv_joint-49-29400'
        self.stgcn_action_classifier.model = None
        self.stgcn_action_classifier.ort_session = None
        self.stgcn_action_classifier.load(model_saved_path, model_name)
        self.stgcn_action_classifier.optimize()
        self.assertIsNotNone(self.stgcn_action_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.stgcn_action_classifier.ort_session = None


if __name__ == "__main__":
    unittest.main()
