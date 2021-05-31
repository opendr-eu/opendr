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
# Copyright (c) 2020 Daniel Honerkamp
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
import functools
import numpy as np
import os
import rospy
from pathlib import Path
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac import SAC
from typing import Optional
from urllib.request import urlretrieve

from control.mobile_manipulation.mobileRL.evaluation import evaluation_rollout
from control.mobile_manipulation.mobileRL.stablebl_callbacks import MobileRLEvalCallback
from engine.constants import OPENDR_SERVER_URL
from engine.learners import LearnerRL


class MobileRLLearner(LearnerRL):
    def __init__(self, env, lr=1e-5, iters=1_000_000, batch_size=64, lr_schedule='linear',
                 lr_end: float = 1e-6, backbone='MlpPolicy', checkpoint_after_iter=20_000, checkpoint_load_iter=0,
                 restore_model_path: Optional[str] = None, temp_path='', device='cuda', seed: int = None,
                 buffer_size: int = 100_000, learning_starts: int = 0, tau: float = 0.001, gamma: float = 0.99,
                 explore_noise: float = 0.5, explore_noise_type='normal', ent_coef='auto', nr_evaluations: int = 50,
                 evaluation_frequency: int = 20_000):
        """
        Specifies a soft-actor-critic (SAC) agent that can be trained for mobile manipulation.
        Internally uses Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3).
        """
        super(LearnerRL, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer='adam',
                                        lr_schedule=lr_schedule, backbone=backbone, network_head='',
                                        checkpoint_after_iter=checkpoint_after_iter,
                                        checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                        device=device, threshold=0.0, scale=1.0)
        self.seed = seed
        self.lr_end = lr_end
        self.nr_evaluations = nr_evaluations
        self.evaluation_frequency = evaluation_frequency
        self.stable_bl_agent = self._construct_agent(env=env,
                                                     buffer_size=buffer_size,
                                                     learning_starts=learning_starts,
                                                     tau=tau,
                                                     gamma=gamma,
                                                     explore_noise=explore_noise,
                                                     explore_noise_type=explore_noise_type,
                                                     ent_coef=ent_coef)

        if checkpoint_load_iter:
            if restore_model_path == 'pretrained':
                assert checkpoint_load_iter == 1_000_000, "pretrained models are provided for step 1_000_000"
            else:
                restore_model_path = os.path.join(restore_model_path, f"model_step{checkpoint_load_iter}")
            self.load(restore_model_path)

    @staticmethod
    def _download_pretrained(save_path: str, robot_name: str):
        checkpoint_load_iter = 1_000_000
        filename = f"model_step{checkpoint_load_iter}.zip"
        file_destination = Path(save_path) / filename
        if not file_destination.exists():
            file_destination.parent.mkdir(parents=True, exist_ok=True)
            url = os.path.join(
                OPENDR_SERVER_URL,
                "control",
                "mobile_manipulation",
                robot_name,
                filename)
            urlretrieve(url=url, filename=file_destination)
        return file_destination

    def _get_lr_fn(self):
        def lin_sched(start_lr, min_lr, progress_remaining):
            return min_lr + progress_remaining * (start_lr - min_lr)

        if self.lr_schedule == 'linear':
            assert self.lr_end is not None
            return functools.partial(lin_sched, self.lr, self.lr_end)
        elif self.lr_schedule:
            raise ValueError(self.lr_schedule)
        else:
            return self.lr

    def _construct_agent(self, env, buffer_size: int, learning_starts: int, tau: float, gamma: float,
                         explore_noise: float, explore_noise_type: str, ent_coef):
        if explore_noise:
            if explore_noise_type == 'normal':
                action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                                 sigma=explore_noise * np.ones(env.action_space.shape))
            elif explore_noise_type == 'OU':
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape),
                                                            sigma=explore_noise * np.ones(env.action_space.shape))
            else:
                raise ValueError(f"Unknown action noise {explore_noise_type}")
        else:
            action_noise = None
        return SAC(policy=self.backbone,
                   env=env,
                   learning_rate=self._get_lr_fn(),
                   buffer_size=buffer_size,
                   learning_starts=learning_starts,
                   batch_size=self.batch_size,
                   tau=tau,
                   gamma=gamma,
                   action_noise=action_noise,
                   policy_kwargs={},
                   tensorboard_log=self.temp_path,
                   create_eval_env=False,
                   seed=self.seed,
                   verbose=0,
                   device=self.device,
                   train_freq=1,
                   ent_coef=ent_coef,
                   target_update_interval=1,
                   target_entropy='auto',
                   use_sde=False)

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
        if logging_path == '':
            logging_path = self.temp_path

        if env is not None:
            assert env.action_space == self.stable_bl_agent.env.action_space
            assert env.observation_space == self.stable_bl_agent.env.observation_space
            self.stable_bl_agent.env = env

        rospy.loginfo("Start learning loop")
        if val_env is not None:
            eval_callback = MobileRLEvalCallback(eval_env=val_env,
                                                 n_eval_episodes=self.nr_evaluations,
                                                 eval_freq=self.evaluation_frequency,
                                                 log_path=logging_path,
                                                 best_model_save_path=logging_path,
                                                 checkpoint_after_iter=self.checkpoint_after_iter,
                                                 verbose=verbose if not silent else False)
        else:
            eval_callback = None
        self.stable_bl_agent.learn(total_timesteps=self.iters,
                                   callback=eval_callback,
                                   eval_env=None)

        self.stable_bl_agent.save(os.path.join(logging_path, f'last_model'))

        for e in [env, val_env]:
            if e is not None:
                env.env_method("clear")
        rospy.loginfo("Training finished")

    def eval(self, env, name_prefix='', nr_evaluations: int = None):
        """
        Evaluate the agent on the specified environment.

        :param env: gym.Env, env to evaluate on
        :param name_prefix: str, name prefix for all logged variables
        :param nr_evaluations: int, number of episodes to evaluate over
        :return:
        """
        if nr_evaluations is None:
            nr_evaluations = self.nr_evaluations
        if isinstance(env, VecEnv):
            assert env.num_envs == 1, "You must pass only one environment when using this function"
            env = env.envs[0]

        prefix = ''
        episode_rewards, episode_lengths, metrics, name_prefix = evaluation_rollout(self.stable_bl_agent, env,
                                                                                    nr_evaluations, name_prefix=prefix,
                                                                                    global_step=self.stable_bl_agent.num_timesteps,
                                                                                    verbose=2)
        env.clear()
        return episode_rewards, episode_lengths, metrics, name_prefix

    def save(self, path):
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
        self.stable_bl_agent.save(path)

    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        if path == 'pretrained':
            path = str(self._download_pretrained(self.temp_path, self.stable_bl_agent.env.get_attr('env_name')[0]))
        self.stable_bl_agent = self.stable_bl_agent.load(path, device=self.device, env=self.stable_bl_agent.env,
                                                         tensorboard_log=self.temp_path)

    def infer(self, batch, deterministic: bool = True):
        return self.stable_bl_agent.predict(batch, deterministic=deterministic)

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target_device):
        raise NotImplementedError()
