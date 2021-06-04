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

# MIT License
#
# Copyright (c) 2021 Halil Ibrahim Ugurlu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

import rospy

from pathlib import Path

from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from engine.learners import LearnerRL
from planning.end_to_end_planning.custom_policies.custom_policies import MultiInputPolicy, create_dual_extractor
# from engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve


class EndToEndPlanningRLLearner(LearnerRL):
    def __init__(self, env, lr=1e-5, iters=1_000_000, batch_size=64, lr_schedule='linear', lr_end: float = 1e-6,
                 backbone='MlpPolicy', checkpoint_after_iter=20_000, checkpoint_load_iter=0, temp_path='',
                 device='cuda', seed: int = None, buffer_size: int = 100_000, learning_starts: int = 0,
                 tau: float = 0.001, gamma: float = 0.99, explore_noise: float = 0.5, explore_noise_type='normal',
                 ent_coef='auto', nr_evaluations: int = 50, evaluation_frequency: int = 20_000):
        """
        Specifies a proximal policy optimization (PPO) agent that can be trained for end to end planning for obstacle avoidance.
        Internally uses Stable-Baselines (https://github.com/hill-a/stable-baselines).
        """
        super(EndToEndPlanningRLLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer='adam',
                                                        lr_schedule=lr_schedule, backbone=backbone, network_head='',
                                                        checkpoint_after_iter=checkpoint_after_iter,
                                                        checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                                        device=device, threshold=0.0, scale=1.0)
        print("initiated")
        net_sizes = [64, 64]
        num_natural_feat = 3
        policy_kwargs = dict(extractor=create_dual_extractor(num_natural_feat),
                             net_arch=[dict(vf=net_sizes, pi=net_sizes)])
        self.env = env #DummyVecEnv(env)
        self.agent = PPO2(MultiInputPolicy, self.env, policy_kwargs=policy_kwargs, n_steps=128, verbose=1)

    def fit(self, env=None, val_env=None, logging_path='', silent=False, verbose=True):
        """
        Train the agent on the environment.

        :param env: gym.Env, optional, if specified use this env to train
        :param val_env:  gym.Env, optional, if specified periodically evaluate on this env
        :param logging_path: str, path for logging and checkpointing
        :param silent: bool, disable verbosity
        :param verbose: bool, enable verbosity
        :return:
        """
        print("fit")

    def eval(self, env, name_prefix='', nr_evaluations: int = None):
        """
        Evaluate the agent on the specified environment.

        :param env: gym.Env, env to evaluate on
        :param name_prefix: str, name prefix for all logged variables
        :param nr_evaluations: int, number of episodes to evaluate over
        :return:
        """
        #env = Monitor(env, filename=logdir, allow_early_resets=True)
        #env = DummyVecEnv([lambda: env])
        self.agent.set_env(env)
        obs = env.reset()
        sum_of_rewards = 0
        for i in range(20):
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            sum_of_rewards += rewards
            if dones:
                break
        return sum_of_rewards

    def save(self, path):
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
        self.agent.save(path)

    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        self.agent = PPO2.load(path) # "./log/best_model11940with_mean_rew8.426711449999999.pkl")
        self.agent.policy = MultiInputPolicy
        self.agent.set_env(self.env)

    def infer(self, batch, deterministic: bool = True):
        return self.agent.predict(batch, deterministic=deterministic)

    def reset(self):
        print("reset")
        # raise NotImplementedError()

    def optimize(self, target_device):
        print("optimize")
        # raise NotImplementedError()
