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
from opendr.perception.facial_expression_recognition.\
    landmark_based_facial_expression_recognition.progressive_spatio_temporal_bln_learner \
    import ProgressiveSpatioTemporalBLNLearner
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


PATH_ = \
    './tests/sources/tools/perception/facial_expression_recognition/landmark_based_facial_expression_recognition/' \
    'facial_expression_recognition_temp'
LOG_PATH_ = ''


class TestLandmarkBasedFacialExpressionRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST ProgressiveSpatioTemporalBLNLearner \n*****"
            "*****************************")
        cls.temp_dir = PATH_
        cls.logging_path = LOG_PATH_
        cls.pstbln_facial_expression_classifier = ProgressiveSpatioTemporalBLNLearner(
                                                  device="cpu", temp_path=cls.temp_dir,
                                                  batch_size=5, epochs=1,
                                                  checkpoint_after_iter=1, val_batch_size=5,
                                                  dataset_name='CASIA', experiment_name='pstbln_mcdo_casia',
                                                  blocksize=5, numblocks=2, numlayers=2, topology=[],
                                                  layer_threshold=1e-4, block_threshold=1e-4)
        cls.experiment_name = 'pstbln_mcdo_casia'
        # Download all required files for testing
        cls.Train_DATASET_PATH = cls.pstbln_facial_expression_classifier.download(
            mode="train_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Val_DATASET_PATH = cls.pstbln_facial_expression_classifier.download(
            mode="val_data", path=os.path.join(cls.temp_dir, "data"))
        cls.Test_DATASET_PATH = cls.pstbln_facial_expression_classifier.download(
            mode="test_data", path=os.path.join(cls.temp_dir, "data"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir, "data"))
        rmdir(os.path.join(cls.temp_dir))

    def test_network_builder(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="CASIA")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="CASIA")
        self.pstbln_facial_expression_classifier.topology = []
        topology_before = []
        self.pstbln_facial_expression_classifier.network_builder(dataset=training_dataset,
                                                                 val_dataset=validation_dataset,
                                                                 monte_carlo_dropout=True, mcdo_repeats=3,
                                                                 train_data_filename='train.npy',
                                                                 train_labels_filename='train_labels.pkl',
                                                                 val_data_filename="val.npy",
                                                                 val_labels_filename="val_labels.pkl",
                                                                 verbose=False)
        topology_after = self.pstbln_facial_expression_classifier.topology
        self.assertNotEqual(len(topology_before), len(topology_after),
                            msg="Model topology did not change after running network_builder.")

    def test_fit(self):
        training_dataset = ExternalDataset(path=self.Train_DATASET_PATH, dataset_type="CASIA")
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="CASIA")
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.init_model()
        m = list(self.pstbln_facial_expression_classifier.model.parameters())[0].clone()
        self.pstbln_facial_expression_classifier.fit(dataset=training_dataset, val_dataset=validation_dataset,
                                                     silent=True, verbose=False,
                                                     monte_carlo_dropout=True, mcdo_repeats=3,
                                                     train_data_filename='train.npy',
                                                     train_labels_filename='train_labels.pkl',
                                                     val_data_filename="val.npy",
                                                     val_labels_filename="val_labels.pkl")
        self.assertFalse(torch.equal(m, list(self.pstbln_facial_expression_classifier.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.init_model()
        validation_dataset = ExternalDataset(path=self.Val_DATASET_PATH, dataset_type="CASIA")
        eval_results = self.pstbln_facial_expression_classifier.eval(validation_dataset,
                                                                     monte_carlo_dropout=True, mcdo_repeats=100,
                                                                     verbose=False,
                                                                     silent=True,
                                                                     val_data_filename='val.npy',
                                                                     val_labels_filename='val_labels.pkl')
        self.assertNotEqual(len(eval_results["score"]), 0, msg="Eval results contains empty list.")

    def test_infer(self):
        test_data = np.load(self.Test_DATASET_PATH)[0:1]
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.init_model()
        category = self.pstbln_facial_expression_classifier.infer(test_data)
        self.assertIsNotNone(category.confidence, msg="The predicted confidence score is None")

    def test_save_load(self):
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.ort_session = None
        self.pstbln_facial_expression_classifier.init_model()
        self.pstbln_facial_expression_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                                      model_name='test_pstgcn')
        self.pstbln_facial_expression_classifier.model = None
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                                      model_name='test_pstgcn')
        self.assertIsNotNone(self.pstbln_facial_expression_classifier.model, "model is None after loading pt model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, self.experiment_name))

    def test_optimize(self):
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.ort_session = None
        self.pstbln_facial_expression_classifier.init_model()
        self.pstbln_facial_expression_classifier.optimize()
        self.assertIsNotNone(self.pstbln_facial_expression_classifier.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.pstbln_facial_expression_classifier.ort_session = None
        rmfile(os.path.join(self.temp_dir, self.experiment_name))

    def test_save_load_onnx(self):
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.ort_session = None
        self.pstbln_facial_expression_classifier.init_model()
        self.pstbln_facial_expression_classifier.optimize()
        self.pstbln_facial_expression_classifier.save(path=os.path.join(self.temp_dir, self.experiment_name),
                                                      model_name='onnx_model')
        self.pstbln_facial_expression_classifier.model = None
        self.pstbln_facial_expression_classifier.topology = [1]
        self.pstbln_facial_expression_classifier.load(path=os.path.join(self.temp_dir, self.experiment_name),
                                                      model_name='onnx_model')
        self.assertIsNotNone(self.pstbln_facial_expression_classifier.ort_session,
                             "ort_session is None after loading onnx model.")
        # Cleanup
        self.pstbln_facial_expression_classifier.ort_session = None
        rmdir(os.path.join(self.temp_dir, self.experiment_name))


if __name__ == "__main__":
    unittest.main()
