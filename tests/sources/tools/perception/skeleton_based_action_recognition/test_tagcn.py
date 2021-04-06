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


# LOG_PATH_ = os.path.join(".", "tests", "sources", "tools", "perception", "skeleton_based_action_recognition", "logs")
PATH_ = './tests/sources/tools/perception/skeleton_based_action_recognition'
LOG_PATH_ = ''


class TestSkeletonBasedActionRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.tagcn_action_classifier = STGCNLearner(device="cpu", temp_path=cls.temp_dir, batch_size=1, epochs=1,
                                                   checkpoint_after_iter=1, val_batch_size=1,
                                                   dataset_name='nturgbd_cv', experiment_name='tagcn_nturgbd',
                                                   method_name='tagcn', num_frames=300, num_subframes=100)
        cls.experiment_name = 'tagcn_nturgbd'
        # Download all required files for testing
        cls.Pretrained_MODEL_PATH = cls.tagcn_action_classifier.download(
            mode="pretrained", path=os.path.join(cls.temp_dir, "pretrained_models"))
        cls.Train_DATASET_PATH = cls.tagcn_action_classifier.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Val_DATASET_PATH = cls.tagcn_action_classifier.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Test_DATASET_PATH = cls.tagcn_action_classifier.download(
            mode="test_data", path=os.path.join(cls.temp_dir, "data"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "data"))
        rmdir(os.path.join(cls.temp_dir, "pretrained_models"))

    def test_fit(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.init_model()
        m = list(self.tagcn_action_classifier.model.parameters())[0].clone()
        self.tagcn_action_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset, silent=True,
                                         train_data_filename='train_joints.npy',
                                         train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
                                         val_labels_filename="val_labels.pkl",
                                         skeleton_data_type='joint')
        self.assertFalse(torch.equal(m, list(self.tagcn_action_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'tagcn_nturgbd-0-10'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        score = self.tagcn_action_classifier.eval(validation_dataset, val_data_filename='val_joints.npy',
                                                  val_labels_filename='val_labels.pkl',
                                                  skeleton_data_type='joint')
        self.assertNotEqual(len(score), 0,
                            msg="Eval results contains empty list.")

    def test_multi_stream_eval(self):
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'tagcn_nturgbd-0-10'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        score_joints = self.tagcn_action_classifier.eval(validation_dataset, val_data_filename='val_joints.npy',
                                                         val_labels_filename='val_labels.pkl',
                                                         skeleton_data_type='joint')
        score_bones = self.tagcn_action_classifier.eval(validation_dataset, val_data_filename='val_joints.npy',
                                                        val_labels_filename='val_labels.pkl',
                                                        skeleton_data_type='bone')
        scores = [score_joints, score_bones]
        total_score = self.tagcn_action_classifier.multi_stream_eval(validation_dataset, scores,
                                                                     data_filename='val_joints.npy',
                                                                     labels_filename='val_labels.pkl'
                                                                     )
        self.assertNotEqual(len(total_score), 0, msg="results of multi-stream-eval contains empty list.")

    def test_infer(self):
        test_data = np.load(self.Test_DATASET_PATH)
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'tagcn_nturgbd-0-10'
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        model_output = self.tagcn_action_classifier.infer(test_data)
        self.assertIsNotNone(model_output, msg="The model output is None")

    def test_save_load(self):
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.ort_session = None
        self.tagcn_action_classifier.init_model()
        self.tagcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='test_tagcn')
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='test_tagcn')
        self.assertIsNotNone(self.tagcn_action_classifier.model, "model is None after loading pt model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, self.experiment_name))

    def test_save_load_onnx(self):
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.ort_session = None
        self.tagcn_action_classifier.init_model()
        self.tagcn_action_classifier.optimize()
        self.tagcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='onnx_model_temp')
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                          model_name='onnx_model_temp')
        self.assertIsNotNone(self.tagcn_action_classifier.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        rmfile(os.path.join(self.temp_dir, self.experiment_name, "onnx_model_temp.onnx"))
        rmdir(os.path.join(self.temp_dir, self.experiment_name))
        self.tagcn_action_classifier.ort_session = None

    def test_optimize(self):
        model_saved_path = self.Pretrained_MODEL_PATH
        model_name = 'tagcn_nturgbd-0-10'
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.ort_session = None
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        self.tagcn_action_classifier.optimize()
        self.assertIsNotNone(self.tagcn_action_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.tagcn_action_classifier.ort_session = None
        rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))


if __name__ == "__main__":
    unittest.main()
