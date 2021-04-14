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

import os
from typing import Callable, Union, Optional

import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn')
import rospy

from stable_baselines3.sac import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from mobileRL.stablebl_callbacks import ModulationEvalCallback
from mobileRL.evaluation import evaluation_rollout
from engine.learners import LearnerRL


# TODO:
#   where to define dependencies?
#   test that demo works
#   test that loading pretrained model works
#   test that examples in readme work
#   add a note to the pull request that some launchfiles stem from open-source ROS packages


class MobileRLLearner(LearnerRL):
    def __init__(self, env, lr=1e-5, iters=1_000_000, batch_size=64, lr_schedule='linear',
                 lr_end: float = 1e-6, backbone='MlpPolicy', checkpoint_after_iter=20_000, checkpoint_load_iter=0,
                 checkpoint_path: Optional[str] = None,
                 temp_path='', device='cuda', seed: int = None, buffer_size: int = 100_000, learning_starts: int = 0,
                 tau: float = 0.001, gamma: float = 0.99, explore_noise: float = 0.5, explore_noise_type='normal',
                 ent_coef='auto', nr_evaluations: int = 50, evaluation_frequency: int = 20_000):
        """
        Specifies an soft-actor-critic (SAC) agent that can be trained for mobile manipulation.
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
        self.checkpoint_path = checkpoint_path
        if checkpoint_load_iter:
            self.load(os.path.join(checkpoint_path, f'model_step{checkpoint_load_iter}'))

    @property
    def lr(self):
        def create_lr_schedule(start_lr: float, min_lr: float, total_decay_steps: int) -> Union[float, Callable]:
            """
            Return a linearly decaying lr. total_decay_steps maps from progrss to current step.
            Note: if using random exploration steps, the lr will already decay during those
            progress_remaining: falling from 1 to 0
            """
            if min_lr == -1:
                return start_lr

            def lin_sched(progress_remaining):
                return min_lr + progress_remaining * (start_lr - min_lr)

            assert 0 < start_lr <= 0.1
            assert 0 <= min_lr <= start_lr
            assert total_decay_steps >= 1
            return lambda progress_remaining: lin_sched(progress_remaining)

        if self.lr_schedule == 'linear':
            assert self.lr_end is not None
            return create_lr_schedule(start_lr=self._lr,
                                      min_lr=self.lr_end,
                                      total_decay_steps=1_000_000)
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

        common_args = {
            'policy': self.backbone,
            'env': env,
            'learning_rate': self.lr,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': self.batch_size,
            'tau': tau,
            'gamma': gamma,
            'action_noise': action_noise,
            'policy_kwargs': {},
            'tensorboard_log': self.temp_path,
            'create_eval_env': False,
            'seed': self.seed,
            'verbose': 0,
            'device': self.device,
        }

        common_args.update({"train_freq": 1,
                            "ent_coef": ent_coef,
                            "target_update_interval": 1,
                            "target_entropy": 'auto',
                            "use_sde": False
                            })
        agent = SAC(**common_args)
        return agent

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
        if env is not None:
            assert env.action_space == self.stable_bl_agent.env.action_space
            assert env.observation_space == self.stable_bl_agent.env.observation_space
            self.stable_bl_agent.env = env

        rospy.loginfo("Start learning loop")
        eval_callback = ModulationEvalCallback(eval_env=val_env,
                                               n_eval_episodes=self.nr_evaluations,
                                               eval_freq=self.evaluation_frequency,
                                               log_path=logging_path,
                                               best_model_save_path=self.checkpoint_path,
                                               checkpoint_after_iter=self.checkpoint_after_iter,
                                               verbose=verbose if not silent else False)
        self.stable_bl_agent.learn(total_timesteps=self.iters,
                                   callback=eval_callback,
                                   eval_env=None)

        self.stable_bl_agent.save(os.path.join(eval_callback.best_model_save_path, f'last_model'))

        env.env_method("clear")
        val_env.env_method("clear")
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

        rospy.loginfo(f"Evaluating on task {env.taskname} with {env.world_type} execution.")
        prefix = ''
        evaluation_rollout(self.stable_bl_agent, env, nr_evaluations, name_prefix=prefix,
                           global_step=self.stable_bl_agent.num_timesteps, verbose=2)
        env.clear()

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
        self.stable_bl_agent.load(path)
