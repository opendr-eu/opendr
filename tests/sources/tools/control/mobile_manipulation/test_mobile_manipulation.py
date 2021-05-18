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

import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from control.mobile_manipulation.mobile_manipulation_learner import MobileRLLearner
from control.mobile_manipulation.mobileRL.utils import create_env

TEST_ITERS = 5
TEMP_SAVE_DIR = Path(__file__) / "mobile_manipulation_tmp"

EVAL_ENV_CONFIG = {
    'env': 'pr2',
    'penalty_scaling': 0.01,
    'time_step': 0.02,
    'seed': 42,
    'strategy': 'dirvel',
    'world_type': 'sim',
    'init_controllers': False,
    'perform_collision_check': True,
    'vis_env': False,
    'transition_noise_base': 0.0,
    'ik_fail_thresh': 20,
    'ik_fail_thresh_eval': 100,
    'learn_vel_norm': -1,
    'slow_down_real_exec': 2,
    'head_start': 0,
    'node_handle': 'train_env'
}

class MobileManipulationTest(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        cls.env = create_env(EVAL_ENV_CONFIG, task='rndstartrndgoal', node_handle="train_env", wrap_in_dummy_vec=True, flatten_obs=True)
        cls.learner = MobileRLLearner(cls.env, device="cpu", iters=TEST_ITERS)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_ckpt_download(self):
        ckpt_folder = self.learner._download_pretrained(1_000_000, TEMP_SAVE_DIR, 'pr2')
        self.assertTrue(Path(ckpt_folder).exists, "Checkpoint file could not be downloaded")
        # Remove temporary files
        try:
            shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")

    def test_fit(self):
        weights_before_fit = list(self.learner.stable_bl_agent.get_parameters().clone())
        self.learner.fit()
        self.assertFalse(torch.equal(weights_before_fit, list(self.learner.stable_bl_agent.get_parameters())),
                         msg="Fit method did not alter model weights")

    def test_eval(self):
        episode_rewards, episode_lengths, metrics, name_prefix = self.learner.eval(self.env, nr_evaluations=1)
        self.assertTrue(episode_rewards <= 0.0, "Test reward not below 0.")
        self.assertTrue(episode_lengths >= 0.0, "Test episode lengths is negative")

    def test_infer(self):
        obs = self.env.observation_space.sample()
        actions = self.learner.infer(obs)
        actions = np.array(actions)
        self.assertTrue(actions.shape == self.env.action_space.shape)
        self.assertTrue((actions >= -1).all(), "Actions below -1")
        self.assertTrue((actions <= 1).all(), "Actions above 1")

    def test_save_load(self):
        weights_before_saving = list(self.learner.stable_bl_agent.get_parameters()).clone()
        self.learner.save(TEMP_SAVE_DIR)

        ckpt_file = self.learner._download_pretrained(1_000_000, TEMP_SAVE_DIR, 'pr2')
        self.learner.load(str(ckpt_file))

        self.assertFalse(torch.equal(weights_before_saving, list(self.learner.stable_bl_agent.get_parameters())),
                         msg="Load() did not alter model weights")

        self.learner.load(TEMP_SAVE_DIR)
        self.assertTrue(torch.equal(weights_before_saving, list(self.learner.stable_bl_agent.get_parameters())),
                        msg="Load did not restore initial weights correctly")
        # with open(os.path.join(TEMP_SAVE_DIR, os.path.basename(TEMP_SAVE_DIR) + ".json")) as jsonfile:
        #     metadata = json.load(jsonfile)
        # self.assertTrue(all(key in metadata for key in ["model_paths",
        #                                                 "framework",
        #                                                 "format",
        #                                                 "has_data",
        #                                                 "inference_params",
        #                                                 "optimized",
        #                                                 "optimizer_info"]))

        # Remove temporary files
        try:
            shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")


if __name__ == "__main__":
    unittest.main()
