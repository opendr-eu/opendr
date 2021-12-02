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
import os
import shutil
import unittest
from pathlib import Path

from opendr.planning.end_to_end_planning.e2e_planning_learner import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.envs.agi_env import AgiEnv
import opendr
import gym

TEST_ITERS = 3
TEMP_SAVE_DIR = Path(__file__).parent / "end_to_end_planning_tmp/"
TEMP_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# EVAL_ENV_CONFIG = {
#     'env': 'pr2',
#     'penalty_scaling': 0.01,
#     'time_step': 0.02,
#     'seed': 42,
#     'strategy': 'dirvel',
#     'world_type': 'sim',
#     'init_controllers': False,
#     'perform_collision_check': True,
#     'vis_env': False,
#     'transition_noise_base': 0.0,
#     'ik_fail_thresh': 20,
#     'ik_fail_thresh_eval': 100,
#     'learn_vel_norm': -1,
#     'slow_down_real_exec': 2,
#     'head_start': 0,
#     'node_handle': 'train_env'
# }


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
        cls.env = AgiEnv()
        cls.learner = EndToEndPlanningRLLearner(cls.env)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_infer(self):
        obs = self.env.observation_space.sample()
        action = self.learner.infer(obs)[0]
        self.assertTrue((action >= 0), "Actions below 0")
        self.assertTrue((action < self.env.action_space.n), "Actions above discrete action space dimensions")

    def test_eval(self):
        episode_reward = self.learner.eval(self.env)
        self.assertTrue((episode_reward > -100), "Episode reward cannot be lower than -100")
        self.assertTrue((episode_reward < 100), "Episode reward cannot pass 100")

    def test_eval_pretrained(self):
        self.learner.load(path=Path(opendr.__file__).parent / "planning/end_to_end_planning/pretrained_model/saved_model.pkl")
        episode_reward = self.learner.eval(self.env)
        self.assertTrue((episode_reward > -30), "Episode reward should be higher than -30")

    def test_fit(self):
        self.learner.__init__(self.env, n_steps=12)
        initial_weights = self.learner.agent.get_parameters()
        self.learner.fit(logging_path=str(TEMP_SAVE_DIR), total_timesteps=15)
        trained_weights = self.learner.agent.get_parameters()
        self.assertFalse(isequal_dict_of_ndarray(initial_weights, trained_weights), "Fit method did not change model weights")

    def test_save_load(self):
        self.learner.__init__(self.env)
        initial_weights = self.learner.agent.get_parameters()
        self.learner.save(str(TEMP_SAVE_DIR) + "/init_weights.pkl")
        self.learner.load(path=Path(opendr.__file__).parent / "planning/end_to_end_planning/pretrained_model/saved_model.pkl")
        self.assertFalse(isequal_dict_of_ndarray(initial_weights, self.learner.agent.get_parameters()),
                         "Load method did not change model weights")
        self.learner.load(str(TEMP_SAVE_DIR) + "/init_weights.pkl")
        self.assertTrue(isequal_dict_of_ndarray(initial_weights, self.learner.agent.get_parameters()),
                         "Load method did not load the same model weights")



if __name__ == "__main__":
    unittest.main()