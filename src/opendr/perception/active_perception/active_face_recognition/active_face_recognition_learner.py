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
import os

from urllib.request import urlretrieve
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from opendr.engine.learners import LearnerRL
from opendr.engine.constants import OPENDR_SERVER_URL

from active_face_recognition_env import Env


class ActiveFaceRecognitionLearner(LearnerRL):
    def __init__(self, lr=0.0003, iters=5000000, batch_size=64,
                 checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda',
                 n_steps=6400,
                 gamma=0.9,
                 clip_range=0.1,
                 target_kl=0.1,
                 ):
        super(ActiveFaceRecognitionLearner, self).__init__(lr=lr,
                                                           iters=iters,
                                                           batch_size=batch_size,
                                                           checkpoint_after_iter=checkpoint_after_iter,
                                                           checkpoint_load_iter=checkpoint_load_iter,
                                                           temp_path=temp_path,
                                                           device=device,
                                                           )
        self.n_steps = n_steps
        self.target_kl = target_kl
        self.gamma = gamma
        self.clip_range = clip_range
        self.env = Env()
        self.agent = None

    def fit(self, logging_path='./', verbose=True):
        """
        Train the agent on the environment.

        :param env: gym.Env, optional, if specified use this env to train
        :param logging_path: str, path for logging and checkpointing
        :param verbose: bool, enable verbosity
        """
        self.env = Monitor(self.env, filename=logging_path)
        if verbose:
            verbose = 2
        else:
            verbose = 0
        self.agent = PPO('CnnPolicy', env=self.env,
                         verbose=verbose,
                         n_steps=self.n_steps,
                         learning_rate=self.lr,
                         gamma=self.gamma,
                         tensorboard_log=logging_path,
                         batch_size=self.batch_size,
                         target_kl=self.target_kl,
                         clip_range=self.clip_range,
                         _init_setup_model=True,
                         )
        if self.checkpoint_after_iter > 0:
            checkpoint_callback = CheckpointCallback(save_freq=self.checkpoint_after_iter, save_path=logging_path,
                                                     name_prefix='rl_model')
            self.agent.learn(total_timesteps=self.iters, callback=checkpoint_callback)
        else:
            self.agent.learn(total_timesteps=self.iters)

    def eval(self, num_episodes=10, deterministic=False):
        """
        Evaluate the agent on the specified environment.

        :param deterministic: use deterministic actions from the policy
        :type deterministic: bool
        :param num_episodes: number of episodes to evaluate agent
        :type num_episodes: int
        :return: average rewards through the episodes
        """
        obs = self.env.reset()
        sum_of_rewards = 0
        for i in range(num_episodes):
            action, _states = self.agent.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = self.env.step(action)
            sum_of_rewards += rewards
            if dones:
                break
        avg_rewards = sum_of_rewards / num_episodes
        return {"rewards_collected": avg_rewards}

    def infer(self, observation, deterministic: bool = False):
        """
        :param observation: single observation
        :type observation: engine.Image
        :param deterministic: use deterministic actions from the policy
        :type deterministic: bool
        :return: the selected action
        :rtype: int or list
        """
        return self.agent.predict(observation, deterministic=deterministic)

    def download(self, path=None):
        """
         Downloads a pretrained model to the path provided.

         :param path: Path to download model
         :type path: str
         """
        if path is None:
            path = self.temp_path
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(os.path.join(path, 'active_fr.zip')):
            url = OPENDR_SERVER_URL + 'perception/active_perception/active_face_recognition/'
            url_model = os.path.join(url, 'active_fr.zip')
            urlretrieve(url_model, os.path.join(path, 'active_fr.zip'))
            print('Model downloaded')
        else:
            print('Model already exists')

    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        """
        self.agent = PPO.load(path)

    def save(self, path=None):
        """
        Saves a trained model to the path provided.
        :param path: Path to save model
        :type path: str
        """
        if path is None:
            path = self.temp_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.save(path)

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target_device):
        raise NotImplementedError()
