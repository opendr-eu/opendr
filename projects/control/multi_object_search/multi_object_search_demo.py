# Copyright 2020-2024 OpenDR European Project
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
import os
import random
import torch
from typing import Callable
from opendr.control.multi_object_search.algorithm.SB3.vec_env import VecEnvExt
from stable_baselines3.common.vec_env import VecMonitor
from opendr.control.multi_object_search import MultiObjectSearchRLLearner
from opendr.control.multi_object_search import MultiObjectEnv
from pathlib import Path
from igibson.utils.utils import parse_config
from stable_baselines3.common.utils import set_random_seed

CONFIG_FILE = str(Path(__file__).parent / 'best_defaults.yaml')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_env(config, logpath):
    # Currently only for 8 training processes. Feel free to extend the list.
    train_set = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int']

    # List corresponds to which scenes are oversampled and which are not.

    mix_sample = {'Merom_0_int': False, 'Benevolence_0_int': True, 'Pomaria_0_int': False, 'Wainscott_1_int': False,
                  'Rs_int': True, 'Ihlen_0_int': False, 'Beechwood_1_int': False, 'Ihlen_1_int': False}

    def make_env(rank: int, seed: int = 0, data_set=[]) -> Callable:
        def _init() -> MultiObjectEnv:
            env_ = MultiObjectEnv(
                config_file=CONFIG_FILE,
                scene_id=data_set[rank],
                mix_sample=mix_sample[data_set[rank]]
            )

            env_.seed(seed + rank)

            return env_

        set_random_seed(rank)

        return _init

    num_cpu_train = config.get('num_cpu', 1)

    if config.get('evaluate', False):
        env = MultiObjectEnv(config_file=CONFIG_FILE, scene_id="Benevolence_1_int")
    else:
        env = VecEnvExt([make_env(i, data_set=train_set) for i in range(num_cpu_train)])
        env = VecMonitor(env, filename=logpath)

    return env


def main():

    main_path = Path(__file__).parent
    logpath = f'{main_path}/logs/'

    checkpoint_path = f'{main_path}/checkpoints/'
    config = parse_config(CONFIG_FILE)

    os.makedirs(logpath, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(config.get('seed', 0))

    # create envs
    env = create_env(config, logpath)
    agent = MultiObjectSearchRLLearner(
        env,
        device=device,
        iters=config.get('train_iterations', 500),
        temp_path=checkpoint_path,
        config_filename=CONFIG_FILE)

    # train
    if config.get('evaluate', False):
        print("Start Evaluation")

        eval_scenes = [
            'Benevolence_1_int',
            'Pomaria_2_int',
            'Benevolence_2_int',
            'Wainscott_0_int',
            'Beechwood_0_int',
            'Pomaria_1_int',
            'Merom_1_int']

        agent.load("pretrained")

        deterministic_policy = config.get('deterministic_policy', False)

        for scene in eval_scenes:
            metrics = agent.eval(
                env,
                name_prefix='Multi_Object_Search',
                name_scene=scene,
                nr_evaluations=75,
                deterministic_policy=deterministic_policy)

            print(f"Success-rate for {scene} : {metrics['metrics']['success']} \n\
                SPL for {scene} : {metrics['metrics']['spl']}")
    else:
        agent.fit(env)


if __name__ == '__main__':
    main()
