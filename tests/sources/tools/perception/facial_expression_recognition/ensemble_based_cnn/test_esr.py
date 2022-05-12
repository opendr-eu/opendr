# Copyright 2020-2022 OpenDR European Project
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
from opendr.perception.facial_expression_recognition import EnsembleCNNLearner
from opendr.engine.data import Image
from opendr.perception.facial_expression_recognition import datasets
from os import path, makedirs
from torch.utils.data import DataLoader


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


PATH_ = './temp'
DATA_PATH = '../../FER_data/AffectNet'


class TestEnsembleBasedCNNLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST Ensemble Based CNN Learner for Facial Expression and "
            "Emotion Analysis\n*"
            "*********************************")
        if not path.isdir(PATH_):
            makedirs(PATH_)
        cls.temp_dir = PATH_

        cls.learner = EnsembleCNNLearner(device="cpu", temp_path=cls.temp_dir,
                                         batch_size=2, max_training_epoch=1, ensemble_size=9,
                                         name_experiment='esr_test', base_path_experiment=PATH_,
                                         lr=1e-1, categorical_train=True, dimensional_finetune=True,
                                         base_path_to_dataset=DATA_PATH)

        cls.dataset_path = cls.learner.base_path_to_dataset
        cls.Pretrained_MODEL_PATH = './trained_models/esr_9'

        # Download all required files for testing

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(os.path.join(cls.temp_dir))

    '''def test_fit(self):
        print("\n\n**********************************\nTest ESR fit function \n*"
              "*********************************")

        self.learner.model = None
        self.learner.init_model(num_branches=self.learner.ensemble_size)

        m = list(self.learner.model.parameters())[0].clone()
        self.learner.fit()
        self.assertFalse(torch.equal(m, list(self.learner.model.parameters())[0]),
                         msg="Model parameters did not change after running fit.")

    def test_eval(self):
        print("\n\n**********************************\nTest ESR eval function \n*"
              "*********************************")
        path_to_saved_network = self.Pretrained_MODEL_PATH
        self.learner.load(self.learner.ensemble_size, path_to_saved_network=path_to_saved_network, fix_backbone=True)
        if self.learner.categorical_train:
            eval_categorical_results = self.learner.eval(eval_type='categorical')
        if self.learner.dimensional_finetune:
            eval_dimensional_results = self.learner.eval(eval_type='dimensional')

        self.assertNotEqual(sum([len(eval_dimensional_results["valence_arousal_losses"][i]) for i in range(2)]), 0,
                            msg="Eval results contains empty lists for valence and arousal estimation loss")
        self.self.assertNotEqual(sum(eval_categorical_results['running_emotion_loss']), 0.0,
                                 msg="Eval results have zero loss for categorical expression recognition")

    def test_infer(self):
        print("\n\n**********************************\nTest ESR infer function \n*"
              "*********************************")
        path_to_saved_network = self.Pretrained_MODEL_PATH
        val_data = datasets.AffectNetCategorical(idx_set=2,
                                                 max_loaded_images_per_label=2,
                                                 transforms=None,
                                                 is_norm_by_mean_std=False,
                                                 base_path_to_affectnet=self.learner.base_path_to_dataset)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=8)
        batch = next(iter(val_loader))[0]
        self.learner.model = None
        self.learner.load(self.learner.ensemble_size, path_to_saved_network=path_to_saved_network, fix_backbone=True)
        # input is Tensor
        ensemble_emotion_results, ensemble_dimension_results = self.learner.infer(batch)
        self.assertIsNotNone(ensemble_emotion_results[0].confidence, msg="The predicted confidence score is None")
        self.self.assertNotEqual((sum(sum(ensemble_dimension_results))).numpy(), 0.0,
                                 msg="overall ensembled dimension results are zero")

        # Input is Image
        ensemble_emotion_results, ensemble_dimension_results = self.learner.infer(Image(batch[0]))
        self.assertIsNotNone(ensemble_emotion_results[0].confidence, msg="The predicted confidence score is None")
        self.self.assertNotEqual((sum(sum(ensemble_dimension_results))).numpy(), 0.0,
                                 msg="overall ensembled dimension results are zero")

        # Input is List[Image]
        ensemble_emotion_results, ensemble_dimension_results = self.learner.infer([Image(v) for v in batch])
        self.assertIsNotNone(ensemble_emotion_results[0].confidence, msg="The predicted confidence score is None")
        self.self.assertNotEqual((sum(sum(ensemble_dimension_results))).numpy(), 0.0,
                                 msg="overall ensembled dimension results are zero")'''

    def test_save_load(self):
        print("\n\n**********************************\nTest ESR save_load function \n*"
              "*********************************")
        path_to_saved_network = path.join(self.temp_dir, self.learner.name_experiment)
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.init_model(num_branches=1)
        self.learner.save(state_dicts=self.learner.model.to_state_dict(),
                          base_path_to_save_model=path_to_saved_network,
                          current_branch_save=0)
        self.learner.load(ensemble_size=1, path_to_saved_network=path_to_saved_network, fix_backbone=True)
        self.assertIsNotNone(self.learner.model, "model is None after loading pt model.")
        # Cleanup

    '''def test_save_load_onnx(self):
        print("\n\n**********************************\nTest ESR save_load ONNX function \n*"
              "*********************************")
        path_to_saved_network = path.join(self.temp_dir, self.learner.name_experiment, 'onnx_optimize')
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.init_model(num_branches=1)
        self.learner.optimize()
        self.learner.save(state_dicts=self.learner.model.to_state_dict(),
                          base_path_to_save_model=path_to_saved_network, current_branch_save=0)
        self.learner.model = None
        self.learner.load(ensemble_size=1, path_to_saved_network=path_to_saved_network, fix_backbone=True)
        self.assertIsNotNone(self.learner.ort_session, "ort_session is None after loading onnx model.")
        # Cleanup
        self.learner.ort_session = None

    def test_optimize(self):
        print("\n\n**********************************\nTest ESR optimize function \n*"
              "*********************************")
        path_to_saved_network = self.Pretrained_MODEL_PATH
        self.learner.model = None
        self.learner.ort_session = None
        self.learner.init_model(num_branches=1)
        self.learner.load(ensemble_size=1, path_to_saved_network=path_to_saved_network, fix_backbone=True)
        self.learner.optimize()
        self.assertIsNotNone(self.learner.ort_session,
                             "ort_session is None after optimizing the pretrained model.")
        # Cleanup
        self.learner.ort_session = None'''


if __name__ == "__main__":
    unittest.main()
