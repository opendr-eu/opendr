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
            "\n\n**********************************\nTEST Skeleton Based Action Recognition Learner with TAGCN model\n*"
            "*********************************")
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.tagcn_action_classifier = SpatioTemporalGCNLearner(device=device, temp_path=cls.temp_dir,
                                                               batch_size=2, epochs=1,
                                                               checkpoint_after_iter=1, val_batch_size=2,
                                                               dataset_name='nturgbd_cv',
                                                               num_class=60, num_point=25, num_person=2, in_channels=3,
                                                               graph_type='ntu',
                                                               experiment_name='tagcn_nturgbd_cv_joint',
                                                               method_name='tagcn', num_frames=300, num_subframes=100)
        cls.experiment_name = 'tagcn_nturgbd_cv_joint'
        # Download all required files for testing
        cls.Pretrained_MODEL_PATH_J = cls.tagcn_action_classifier.download(
            path=os.path.join(cls.temp_dir, "pretrained_models", "tagcn"), method_name="tagcn", mode="pretrained",
            file_name='tagcn_nturgbd_cv_joint-49-29400')
        cls.Pretrained_MODEL_PATH_B = cls.tagcn_action_classifier.download(
            path=os.path.join(cls.temp_dir, "pretrained_models", "tagcn"), method_name="tagcn", mode="pretrained",
            file_name='tagcn_nturgbd_cv_bone-49-29400')

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
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.init_model()
        m = list(self.tagcn_action_classifier.model.parameters())[0].clone()
        self.tagcn_action_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset, silent=True,
                                         verbose=False, train_data_filename='train_joints.npy',
                                         train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
                                         val_labels_filename="val_labels.pkl",
                                         skeleton_data_type='joint')
        self.assertFalse(torch.equal(m, list(self.tagcn_action_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'tagcn_nturgbd_cv_joint-49-29400'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        eval_results = self.tagcn_action_classifier.eval(validation_dataset, verbose=False,
                                                         val_data_filename='val_joints.npy',
                                                         val_labels_filename='val_labels.pkl',
                                                         skeleton_data_type='joint')
        self.assertNotEqual(len(eval_results["score"]), 0,
                            msg="Eval results contains empty list.")

    def test_multi_stream_eval(self):
        model_saved_path_joint = self.Pretrained_MODEL_PATH_J
        model_saved_path_bone = self.Pretrained_MODEL_PATH_B
        model_name_joint = 'tagcn_nturgbd_cv_joint-49-29400'
        model_name_bone = 'tagcn_nturgbd_cv_bone-49-29400'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")

        self.tagcn_action_classifier.load(model_saved_path_joint, model_name_joint)
        eval_results_joint = self.tagcn_action_classifier.eval(validation_dataset, verbose=False,
                                                               val_data_filename='val_joints.npy',
                                                               val_labels_filename='val_labels.pkl',
                                                               skeleton_data_type='joint')

        self.tagcn_action_classifier.load(model_saved_path_bone, model_name_bone)
        eval_results_bone = self.tagcn_action_classifier.eval(validation_dataset, verbose=False,
                                                              val_data_filename='val_joints.npy',
                                                              val_labels_filename='val_labels.pkl',
                                                              skeleton_data_type='bone')
        score_joints = eval_results_joint["score"]
        score_bones = eval_results_bone["score"]
        scores = [score_joints, score_bones]
        total_score = self.tagcn_action_classifier.multi_stream_eval(validation_dataset, scores,
                                                                     data_filename='val_joints.npy',
                                                                     labels_filename='val_labels.pkl'
                                                                     )
        self.assertNotEqual(len(total_score), 0, msg="results of multi-stream-eval contains empty list.")

    def test_infer(self):
        test_data = np.load(self.Test_DATASET_PATH)[0:1]
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'tagcn_nturgbd_cv_joint-49-29400'
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        category = self.tagcn_action_classifier.infer(test_data)
        self.assertIsNotNone(category.confidence, msg="The predicted confidence score is None")

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
        self.tagcn_action_classifier.ort_session = None

    def test_optimize(self):
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'tagcn_nturgbd_cv_joint-49-29400'
        self.tagcn_action_classifier.model = None
        self.tagcn_action_classifier.ort_session = None
        self.tagcn_action_classifier.load(model_saved_path, model_name)
        self.tagcn_action_classifier.optimize()
        self.assertIsNotNone(self.tagcn_action_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.tagcn_action_classifier.ort_session = None


if __name__ == "__main__":
    unittest.main()
