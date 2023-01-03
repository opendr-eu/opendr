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
import numpy as np
import gym
import os
from pathlib import Path
from urllib.request import urlretrieve

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from opendr.engine.learners import LearnerRL
from opendr.engine.constants import OPENDR_SERVER_URL


class EndToEndPlanningRLLearner(LearnerRL):
    def __init__(self, env=None, lr=3e-4, n_steps=1024, iters=int(1e5), batch_size=64, checkpoint_after_iter=500,
                 temp_path='', device='cuda'):
        """
        Specifies a proximal policy optimization (PPO) agent that can be trained for end to end planning for obstacle avoidance.
        Internally uses Stable-Baselines 3 (https://github.com/DLR-RM/stable-baselines3.git).
        """
        super(EndToEndPlanningRLLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer='adam',
                                                        network_head='', temp_path=temp_path,
                                                        checkpoint_after_iter=checkpoint_after_iter,
                                                        device=device, threshold=0.0, scale=1.0)
        self.env = env
        self.n_steps = n_steps
        if self.env is None:
            self.agent = PPO.load(os.environ.get(
                "OPENDR_HOME") + '/src/opendr/planning/end_to_end_planning/pretrained_model/saved_model.zip')
            print("Learner is initiated with pretrained model without a gym model.")
        else:
            if isinstance(self.env, DummyVecEnv):
                self.env = self.env.envs[0]
            self.env = DummyVecEnv([lambda: self.env])
            self.agent = PPO("MultiInputPolicy", self.env, learning_rate=self.lr, n_steps=self.n_steps,
                             batch_size=self.batch_size, verbose=1)

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

    def fit(self, env=None, logging_path='', verbose=True):
        """
        Train the agent on the environment.

        :param env: gym.Env, optional, if specified use this env to train
        :param logging_path: str, path for logging and checkpointing
        :param verbose: bool, enable verbosity
        """
        if env is not None:
            if isinstance(env, gym.Env):
                if isinstance(self.env, gym.Env):
                    self.env = env
                else:
                    self.env = env
                    self.agent = PPO("MultiInputPolicy", self.env, learning_rate=self.lr, n_steps=self.n_steps,
                                     batch_size=self.batch_size, verbose=verbose)
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
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=self.logdir, name_prefix='rl_model')
        self.agent.learn(total_timesteps=self.iters, callback=checkpoint_callback)

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
        return {"rewards_collected": sum_of_rewards}

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
        if isinstance(self.env, gym.Env):
            self.agent.set_env(self.env)

    def infer(self, batch, deterministic: bool = True):
        """
        Loads a model from the path provided.

        :param batch: single or list of observations
        :type batch: dict ot list of dict
        :param deterministic: use deterministic actions from the policy
        :type deterministic: bool
        :return: the selected action
        :rtype: int or list
        """
        if isinstance(batch, dict):
            return self.agent.predict(batch, deterministic=deterministic)
        elif isinstance(batch, list) or isinstance(batch, np.ndarray):
            return [self.agent.predict(obs, deterministic=deterministic) for obs in batch]
        else:
            raise ValueError()

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target_device):
        raise NotImplementedError()
