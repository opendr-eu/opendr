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
from optuna.study.study import Study
from opendr.utils.hyperparameter_tuner.dummy_learner import DummyLearner
from opendr.utils.hyperparameter_tuner.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Hyperparameter Tuner\n"
              "**********************************")
        cls.tuner = HyperparameterTuner(DummyLearner)

    def test_optimize(self):
        fit_arguments = {'dataset': None}
        eval_arguments = {'dataset': None}
        timeout = 3.0

        best_params = self.tuner.optimize(
            fit_arguments=fit_arguments,
            eval_arguments=eval_arguments,
            timeout=timeout,
        )
        study = self.tuner.study
        for key, value in study.best_params.items():
            self.assertEqual(best_params[key], value)

    def test_study(self):
        study = self.tuner.study
        self.assertTrue(type(study) is Study)


if __name__ == '__main__':
    unittest.main()
