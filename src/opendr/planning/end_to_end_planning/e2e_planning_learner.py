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
import gym
import os
from pathlib import Path
from urllib.request import urlretrieve

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

from opendr.engine.learners import LearnerRL
from opendr.engine.constants import OPENDR_SERVER_URL

__all__ = ["rospy", ]


class EndToEndPlanningRLLearner(LearnerRL):
    def __init__(self, env, n_steps=128, lr=1e-5, iters=1_000_000, batch_size=64, lr_schedule='linear',
                 lr_end: float = 1e-6, backbone='MlpPolicy', checkpoint_after_iter=20_000, checkpoint_load_iter=0,
                 temp_path='', device='cuda'):
        """
        Specifies a proximal policy optimization (PPO) agent that can be trained for end to end planning for obstacle avoidance.
        Internally uses Stable-Baselines (https://github.com/hill-a/stable-baselines).
        """
        super(EndToEndPlanningRLLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer='adam',
                                                        lr_schedule=lr_schedule, backbone=backbone, network_head='',
                                                        checkpoint_after_iter=checkpoint_after_iter,
                                                        checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                                        device=device, threshold=0.0, scale=1.0)
        self.env = env
        if isinstance(self.env, DummyVecEnv):
            self.env = self.env.envs[0]
        self.env = DummyVecEnv([lambda: self.env])
        self.agent = PPO("MultiInputPolicy", self.env, n_steps=n_steps, verbose=1)

    def download(self, path=None,
                 url=OPENDR_SERVER_URL + "planning/end_to_end_planning"):
        if path is None:
            path = "./end_to_end_planning_tmp/"
        filename = "ardupilot.zip"
        file_destination = Path(path) / filename
        if not file_destination.exists():
            file_destination.parent.mkdir(parents=True, exist_ok=True)
            url = os.path.join(url, filename)
            urlretrieve(url=url, filename=file_destination)
        return file_destination

    def fit(self, env=None, logging_path='', total_timesteps=int(5e4), silent=False, verbose=True):
        """
        Train the agent on the environment.

        :param env: gym.Env, optional, if specified use this env to train
        :param logging_path: str, path for logging and checkpointing
        :param total_timesteps: int, total timesteps to be trained
        :param silent: bool, disable verbosity
        :param verbose: bool, enable verbosity
        :return:
        """
        if env is not None:
            if isinstance(env, gym.Env):
                self.env = env
            else:
                print('env should be gym.Env')
                return
        self.last_checkpoint_time_step = 0
        self.logdir = logging_path
        if isinstance(self.env, DummyVecEnv):
            self.env = self.env.envs[0]
        if isinstance(self.env, Monitor):
            self.env = self.env.env
        self.env = Monitor(self.env, filename=self.logdir)
        self.env = DummyVecEnv([lambda: self.env])
        self.agent.set_env(self.env)
        self.agent.learn(total_timesteps=total_timesteps, callback=self.callback)

    def eval(self, env):
        """
        Evaluate the agent on the specified environment.

        :param env: gym.Env, env to evaluate on
        :return: sum of rewards through the episode
        """
        if isinstance(env, DummyVecEnv):
            env = env.envs[0]
        if isinstance(env, Monitor):
            env = env.env
        # env = Monitor(env, filename=self.logdir)
        env = DummyVecEnv([lambda: env])
        self.agent.set_env(env)
        obs = env.reset()
        sum_of_rewards = 0
        for i in range(50):
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
        self.agent = PPO.load(path)
        # self.agent.policy = MultiInputPolicy
        self.agent.set_env(self.env)

    def infer(self, batch, deterministic: bool = True):
        """
        Loads a model from the path provided.

        :param batch: Path to saved model
        :type batch: list
        :param deterministic: use deterministic actions from the policy
        :type deterministic: bool
        :return: the selected action
        :rtype: int
        """
        return self.agent.predict(batch, deterministic=deterministic)

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target_device):
        raise NotImplementedError()

    def callback(self, _locals, _globals):
        x, y = ts2xy(load_results(self.logdir), 'timesteps')

        if len(y) > 20:
            mean_reward = np.mean(y[-20:])
        else:
            return True

        if x[-1] - self.last_checkpoint_time_step > 20:
            self.last_checkpoint_time_step = x[-1]
            check_point_path = Path(self.logdir,
                                    'checkpoint_save' + str(x[-1]) + 'with_mean_rew' + str(mean_reward))
            self.save(str(check_point_path))

        return True
