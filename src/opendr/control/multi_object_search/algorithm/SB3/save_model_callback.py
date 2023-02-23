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
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        print("REMAINING LR :", progress_remaining * initial_value)
        return progress_remaining * initial_value

    return func


class SaveModel(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str = './', verbose=1, model_id=1):
        super(SaveModel, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.curriculum_stage = 0
        self.save_path = os.path.join(log_dir, 'best_model_retrain_latest_{}'.format(self.curriculum_stage))
        self.model_id = model_id
        self.save_path_last = os.path.join(log_dir, 'last_model')
        self.best_mean_reward = -np.inf
        self.backward_episodes = 100

    def _on_step(self) -> bool:
        if (self.n_calls + 1) % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.backward_episodes:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - \
                        Last mean reward per episode: {mean_reward:.2f}")
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                print(f"Saving new LATEST model to {self.save_path}.zip")
                self.model.save(self.save_path_last)
        return True
