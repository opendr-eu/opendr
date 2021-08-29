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

import gym
import numpy as np
import os
import warnings
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from typing import Union, Optional

from opendr.control.mobile_manipulation.mobileRL.evaluation import evaluation_rollout


class MobileRLEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 debug: bool = False,
                 prefix: str = 'eval',
                 checkpoint_after_iter: int = 0
                 ):
        super(MobileRLEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.checkpoint_after_iter = checkpoint_after_iter

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self._best_mean_reward = - np.infty
        self.log_path = log_path

        self.debug = debug
        self.prefix = prefix

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type"
                          f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def do_eval(self):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)

        eval_env = self.eval_env
        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment when using this function"
            eval_env = eval_env.envs[0]

        return evaluation_rollout(self.model, env=eval_env,
                                  num_eval_episodes=self.n_eval_episodes,
                                  global_step=self.num_timesteps,
                                  name_prefix=self.prefix)

    def _on_step(self) -> bool:
        continue_train = True

        if self.checkpoint_after_iter and self.n_calls and (self.n_calls % self.checkpoint_after_iter == 0) and (
                self.best_model_save_path is not None):
            self.model.save(os.path.join(self.best_model_save_path, f'model_step{self.num_timesteps}'))

        if (self.n_calls == 1) or (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            episode_rewards, episode_lengths, metrics, name_prefix = self.do_eval()

            mean_reward = np.mean(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if self.verbose > 0:
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            if mean_reward > self._best_mean_reward:
                if self.best_model_save_path is not None:
                    print("Saving best model")
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
                self._best_mean_reward = mean_reward

        return continue_train
