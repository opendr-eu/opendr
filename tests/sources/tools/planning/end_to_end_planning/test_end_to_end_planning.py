# Copyright 1996-2020 OpenDR European Project
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
import unittest
from pathlib import Path
from gym.spaces import Box

from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner, UAVDepthPlanningEnv
import opendr
import torch
import os

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

TEST_ITERS = 3
TEMP_SAVE_DIR = Path(__file__).parent / "end_to_end_planning_tmp/"
TEMP_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_first_weight(learner):
    return list(learner.stable_bl_agent.get_parameters()['policy'].values())[0].clone()


def isequal_dict_of_ndarray(first, second):
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)


class EndToEndPlanningTest(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        cls.env = UAVDepthPlanningEnv()
        cls.learner = EndToEndPlanningRLLearner(cls.env, device=device)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_infer(self):
        obs = self.env.observation_space.sample()
        action = self.learner.infer(obs)[0]
        if isinstance(self.env.action_space, Box):
            self.assertTrue((np.abs(action[0]) <= 1), "Action not between -1 and 1")
            self.assertTrue((np.abs(action[1]) <= 1), "Action not between -1 and 1")
        else:
            self.assertTrue((action >= 0), "Actions below 0")
            self.assertTrue((action < self.env.action_space.n), "Actions above discrete action space dimensions")

    def test_eval(self):
        episode_reward = self.learner.eval(self.env)["rewards_collected"]
        self.assertTrue((episode_reward > -100), "Episode reward cannot be lower than -100")
        self.assertTrue((episode_reward < 100), "Episode reward cannot pass 100")

    def test_fit(self):
        self.learner.__init__(self.env, n_steps=12, iters=15)
        initial_weights = self.learner.agent.get_parameters()
        self.learner.fit(logging_path=str(TEMP_SAVE_DIR))
        trained_weights = self.learner.agent.get_parameters()
        self.assertFalse(isequal_dict_of_ndarray(initial_weights, trained_weights),
                         "Fit method did not change model weights")

    def test_save_load(self):
        self.learner.__init__(self.env)
        initial_weights = list(self.learner.agent.get_parameters()['policy'].values())[0].clone()
        self.learner.save(str(TEMP_SAVE_DIR) + "/init_weights")
        self.learner.load(
            path=Path(opendr.__file__).parent / "planning/end_to_end_planning/pretrained_model/saved_model")
        self.assertFalse(
            torch.equal(initial_weights, list(self.learner.agent.get_parameters()['policy'].values())[0].clone()),
            "Load method did not change model weights")
        self.learner.load(str(TEMP_SAVE_DIR) + "/init_weights")
        self.assertTrue(
            torch.equal(initial_weights, list(self.learner.agent.get_parameters()['policy'].values())[0].clone()),
            "Load method did not load the same model weights")


if __name__ == "__main__":
    unittest.main()
