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


from typing import Callable, List, Optional

import gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs


class VecEnvExt(SubprocVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super(VecEnvExt, self).__init__(env_fns, start_method)
        self.succ_rate = [[0] for _ in range(len(env_fns))]
        self.collision_rate = [[0] for _ in range(len(env_fns))]
        self.aux_episode_succ = [[0] for _ in range(len(env_fns))]
        self.aux_episode_coll_rate = [[0] for _ in range(len(env_fns))]
        self.scene_succ = {'Merom_0_int': [], 'Benevolence_0_int': [], 'Pomaria_0_int': [], 'Wainscott_1_int': [],
                           'Rs_int': [], 'Ihlen_0_int': [], 'Beechwood_1_int': [], 'Ihlen_1_int': []}
        n_envs = len(env_fns)
        self.numb_envs = n_envs  # len(env_fns)

    def step_wait(self) -> VecEnvStepReturn:

        results = [remote.recv() for remote in self.remotes]

        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        for i, inf in enumerate(infos):
            if (dones[i]):

                if (len(self.succ_rate[i]) > 48):
                    self.succ_rate[i].pop(0)

                if (len(self.aux_episode_succ[i]) > 48):
                    self.aux_episode_succ[i].pop(0)

                if inf['success'] and inf["aux_episode"]:
                    self.aux_episode_succ[i].append(1)

                elif not inf['success'] and inf["aux_episode"]:
                    self.aux_episode_succ[i].append(0)

                if inf['success']:
                    self.succ_rate[i].append(1)
                else:
                    self.succ_rate[i].append(0)

                    # keep history of 35 episodes in buffer
                if (len(self.collision_rate[i]) > 48):
                    self.collision_rate[i].pop(0)

                self.collision_rate[i].append(inf['collision_step'])

        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
