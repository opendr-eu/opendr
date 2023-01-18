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
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
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
            "\n\n**********************************\nTEST Progressive Skeleton Based Action Recognition Learner \n*****"
            "*****************************")
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.pstgcn_action_classifier = ProgressiveSpatioTemporalGCNLearner(
                                       device=device, temp_path=cls.temp_dir,
                                       batch_size=2, epochs=1,
                                       checkpoint_after_iter=1, val_batch_size=2,
                                       dataset_name='nturgbd_cv',
                                       num_class=60, num_point=25, num_person=2, in_channels=3,
                                       graph_type='ntu',
                                       experiment_name='pstgcn_nturgbd_cv_joint',
                                       blocksize=20, numblocks=2, numlayers=2, topology=[],
                                       layer_threshold=1e-4, block_threshold=1e-4)
        cls.experiment_name = 'pstgcn_nturgbd_cv_joint'
        # Download all required files for testing
        cls.Pretrained_MODEL_PATH_J = cls.pstgcn_action_classifier.download(
            path=os.path.join(cls.temp_dir, "pretrained_models", "pstgcn"), method_name="pstgcn", mode="pretrained",
            file_name='pstgcn_nturgbd_cv_joint-8-4')
        cls.Pretrained_MODEL_PATH_B = cls.pstgcn_action_classifier.download(
            path=os.path.join(cls.temp_dir, "pretrained_models", "pstgcn"), method_name="pstgcn", mode="pretrained",
            file_name='pstgcn_nturgbd_cv_bone-8-4')
        cls.Train_DATASET_PATH = cls.pstgcn_action_classifier.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Val_DATASET_PATH = cls.pstgcn_action_classifier.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Test_DATASET_PATH = cls.pstgcn_action_classifier.download(
            mode="test_data", path=os.path.join(cls.temp_dir, "data"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "data"))
        rmdir(os.path.join(cls.temp_dir, "pretrained_models"))
        rmdir(os.path.join(cls.temp_dir))

    def test_network_builder(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.pstgcn_action_classifier.topology = []
        topology_before = []
        self.pstgcn_action_classifier.network_builder(dataset=training_dataset, val_dataset=validation_dataset,
                                                      train_data_filename='train_joints.npy',
                                                      train_labels_filename='train_labels.pkl',
                                                      val_data_filename="val_joints.npy",
                                                      val_labels_filename="val_labels.pkl",
                                                      skeleton_data_type='joint')
        topology_after = self.pstgcn_action_classifier.topology
        self.assertNotEqual(len(topology_before), len(topology_after),
                            msg="Model topology did not change after running network_builder.")

    def test_fit(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="NTURGBD")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.pstgcn_action_classifier.topology = [1]
        self.pstgcn_action_classifier.init_model()
        m = list(self.pstgcn_action_classifier.model.parameters())[0].clone()
        self.pstgcn_action_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset, silent=True,
                                          verbose=False, train_data_filename='train_joints.npy',
                                          train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
                                          val_labels_filename="val_labels.pkl",
                                          skeleton_data_type='joint')
        self.assertFalse(torch.equal(m, list(self.pstgcn_action_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        model_saved_path = self.Pretrained_MODEL_PATH_J
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        model_name = 'pstgcn_nturgbd_cv_joint-8-4'
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        self.pstgcn_action_classifier.load(model_saved_path, model_name)
        eval_results = self.pstgcn_action_classifier.eval(validation_dataset, verbose=False,
                                                          val_data_filename='val_joints.npy',
                                                          val_labels_filename='val_labels.pkl',
                                                          skeleton_data_type='joint')
        self.assertNotEqual(len(eval_results["score"]), 0, msg="Eval results contains empty list.")

    def test_infer(self):
        test_data = np.load(self.Test_DATASET_PATH)[0:1]
        model_saved_path = self.Pretrained_MODEL_PATH_J
        self.pstgcn_action_classifier.model = None
        model_name = 'pstgcn_nturgbd_cv_joint-8-4'
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.load(model_saved_path, model_name)
        category = self.pstgcn_action_classifier.infer(test_data)
        self.assertIsNotNone(category.confidence, msg="The predicted confidence score is None")

    def test_save_load(self):
        self.pstgcn_action_classifier.topology = [1]
        self.pstgcn_action_classifier.ort_session = None
        self.pstgcn_action_classifier.init_model()
        self.pstgcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                           model_name='test_pstgcn')
        self.pstgcn_action_classifier.model = None
        self.pstgcn_action_classifier.topology = [1]
        self.pstgcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                           model_name='test_pstgcn')
        self.assertIsNotNone(self.pstgcn_action_classifier.model, "model is None after loading pt model.")
        # Cleanup

    def test_multi_stream_eval(self):
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="NTURGBD")
        model_saved_path_joint = self.Pretrained_MODEL_PATH_J
        model_saved_path_bone = self.Pretrained_MODEL_PATH_B
        model_name_joint = 'pstgcn_nturgbd_cv_joint-8-4'
        model_name_bone = 'pstgcn_nturgbd_cv_bone-8-4'

        self.pstgcn_action_classifier.model = None
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.load(model_saved_path_joint, model_name_joint)
        eval_results_joint = self.pstgcn_action_classifier.eval(validation_dataset, verbose=False,
                                                                val_data_filename='val_joints.npy',
                                                                val_labels_filename='val_labels.pkl',
                                                                skeleton_data_type='joint')
        self.pstgcn_action_classifier.model = None
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.load(model_saved_path_bone, model_name_bone)
        eval_results_bone = self.pstgcn_action_classifier.eval(validation_dataset, verbose=False,
                                                               val_data_filename='val_joints.npy',
                                                               val_labels_filename='val_labels.pkl',
                                                               skeleton_data_type='bone')
        score_joints = eval_results_joint["score"]
        score_bones = eval_results_bone["score"]
        scores = [score_joints, score_bones]
        total_score = self.pstgcn_action_classifier.multi_stream_eval(validation_dataset, scores,
                                                                      data_filename='val_joints.npy',
                                                                      labels_filename='val_labels.pkl'
                                                                      )
        self.assertNotEqual(len(total_score), 0, msg="results of multi-stream-eval contains empty list.")

    def test_optimize(self):
        model_saved_path = self.Pretrained_MODEL_PATH_J
        model_name = 'pstgcn_nturgbd_cv_joint-8-4'
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.ort_session = None
        self.pstgcn_action_classifier.init_model()
        self.pstgcn_action_classifier.load(model_saved_path, model_name)
        self.pstgcn_action_classifier.optimize()
        self.assertIsNotNone(self.pstgcn_action_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.pstgcn_action_classifier.ort_session = None

    def test_save_load_onnx(self):
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.ort_session = None
        self.pstgcn_action_classifier.init_model()
        self.pstgcn_action_classifier.optimize()
        self.pstgcn_action_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                           model_name='onnx_model_temp')
        self.pstgcn_action_classifier.model = None
        self.pstgcn_action_classifier.topology = [5, 4, 5, 2, 3, 4, 3, 4]
        self.pstgcn_action_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                           model_name='onnx_model_temp')
        self.assertIsNotNone(self.pstgcn_action_classifier.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        self.pstgcn_action_classifier.ort_session = None


if __name__ == "__main__":
    unittest.main()
