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

import argparse
import numpy as np
import os
import random
import rospy
import sys
import time
import torch
import yaml
from opendr.control.mobile_manipulation import ALL_TASKS
from opendr.control.mobile_manipulation import evaluate_on_task
from opendr.control.mobile_manipulation import create_env
from opendr.control.mobile_manipulation import MobileRLLearner
from pathlib import Path
from stable_baselines3.common.utils import configure_logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(config_path):
    all_tasks = ALL_TASKS.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_best_defaults', type=str2bool, nargs='?', const=True, default=True,
                        help="Replace default values with those from configs/best_defaults.yaml.")
    parser.add_argument('--seed', type=int, default=-1, help="Set to a value >= to use deterministic seed")
    parser.add_argument('--rnd_steps', type=int, default=0,
                        help='Number of random actions to record before starting with rl, subtracted from total_steps')
    parser.add_argument('--total_steps', type=int, default=1_000_000,
                        help='Total number of action/observation steps to take over all episodes')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    #################################################
    # ALGORITHMS
    #################################################
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
    parser.add_argument('--lr_end', type=float, default=1e-6, help="Final / min learning rate. -1 to not decay")
    parser.add_argument('--tau', type=float, default=0.001, help='target value moving average speed')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount')
    parser.add_argument('--explore_noise_type', type=str, default='normal', choices=['normal', 'OU', ''],
                        help='Type of exploration noise')
    parser.add_argument('--explore_noise', type=float, default=0.0, help='')
    parser.add_argument('--ent_coef', default="auto", help="Entropy coefficient. 'auto' to learn it.")
    #################################################
    # Env
    #################################################
    parser.add_argument('--env', type=str.lower, default='pr2', choices=['pr2', 'tiago', 'hsr'], help='')
    parser.add_argument('--task', type=str.lower, default='rndstartrndgoal', choices=all_tasks,
                        help='Train on a specific task env. Might override some other choices.')
    parser.add_argument('--time_step', type=float, default=0.02,
                        help='Time steps at which the RL agent makes decisions during actual execution. NOTE: time_step '
                             'for training is hardcoded in robot_env.cpp.')
    parser.add_argument('--slow_down_real_exec', type=float, default=1.0,
                        help='How much to slow down the planned gripper trajectories during real execun')
    parser.add_argument('--world_type', type=str, default="sim", choices=["sim", "gazebo", "world"],
                        help="What kind of movement execution and where to get updated values from. Sim: analytical "
                             "environemt, don't call controllers, gazebo: gazebo simulator, world: real world")
    parser.add_argument('--strategy', type=str.lower, default="dirvel", choices=["relvelm", "relveld", "dirvel"],
                        help='What velocities to learn: modulate, velocity relative to the gripper velocity, direct base '
                             'velocity')
    parser.add_argument('--ik_fail_thresh', type=int, default=20,
                        help='number of failures after which on it is considered as failed (i.e. failed: '
                             'failures > ik_fail_thresh)')
    parser.add_argument('--ik_fail_thresh_eval', type=int, default=100,
                        help='different eval threshold to make comparable across settings and investigate if it can '
                             'recover from failures')
    parser.add_argument('--penalty_scaling', type=float, default=0.01,
                        help='by how much to scale the penalties to incentivise minimal modulation')
    parser.add_argument('--learn_vel_norm', type=float, default=-1,
                        help="Learn the norm of the next EE-motion. Value is the factor weighting the loss for this. -1 "
                             "to not learn it.")
    parser.add_argument('--perform_collision_check', type=str2bool, nargs='?', const=True, default=True,
                        help='Use the planning scen to perform collision checks (both with environment and self '
                             'collisions)')
    parser.add_argument('--vis_env', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to publish markers to rviz')
    parser.add_argument('--transition_noise_base', type=float, default=0.0,
                        help='Std of Gaussian noise applied to the next base transform during training')
    parser.add_argument('--head_start', type=float, default=0.0,
                        help='Seconds to wait before starting the EE-motion (allowing the base to position itself)')
    #################################################
    # HSR
    #################################################
    parser.add_argument('--hsr_ik_slack_dist', type=float, default=0.1, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_ik_slack_rot_dist', type=float, default=0.05, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_sol_dist_reward', type=str2bool, default=False,
                        help='Penalise distance to perfect ik solution')
    #################################################
    # Eval
    #################################################
    parser.add_argument('--nr_evaluations', type=int, default=50, help='Nr of runs for the evaluation')
    parser.add_argument('--evaluation_frequency', type=int, default=20000, help='In nr of steps')
    parser.add_argument('--evaluation_only', type=str2bool, nargs='?', const=True, default=False,
                        help='If True only model will be loaded and evaluated no training')
    parser.add_argument('--eval_worlds', nargs='+', default=['sim'], choices=['sim', 'gazebo', 'world'],
                        help='Eval execs to run')
    parser.add_argument('--eval_tasks', nargs='+', default=list(all_tasks), choices=all_tasks, help='Eval tasks to run')
    # #################################################
    # Misc
    #################################################
    parser.add_argument('--restore_model_path', type=str, default='pretrained',
                        help='Restore the model and config under this path')
    parser.add_argument('--checkpoint_load_iter', type=int, default=0,
                        help='Restore the model named model_step{x}. See ./model_checkpoints/[robot] for pretrained '
                             'checkpoints. Note: does not restore the config automatically.')
    parser.add_argument('--name', type=str, default="", help='name for this run')

    args = parser.parse_args()
    args = vars(args)

    # user-specified command-line arguments
    cl_args = [k.replace('-', '').split('=')[0] for k in sys.argv]

    if args.pop('load_best_defaults'):
        with open(config_path / 'best_defaults.yaml') as f:
            best_defaults = yaml.safe_load(f)
        # replace with best_default value unless something else was specified through command line
        for k, v in best_defaults[args['env']].items():
            if k not in cl_args:
                args[k] = v

    if args['checkpoint_load_iter']:
        assert args['evaluation_only'], "Continuing to train not supported atm (replay buffer doesn't get saved)"

    args['checkpoint_after_iter'] = args['evaluation_frequency'] if not args['evaluation_only'] else 0

    if args['env'] == 'hsr':
        assert not args['perform_collision_check'], "Collisions can crash due to unsupported geometries"

    # do we need to initialise controllers for the specified tasks?
    tasks_that_need_controllers = [k for k, v in ALL_TASKS.items() if v.requires_simulator()]
    task_needs_gazebo = len(set([args['task']] + args['eval_tasks']).intersection(set(tasks_that_need_controllers))) > 0
    world_type_needs_controllers = set([args['world_type']] + args['eval_worlds']) != {'sim'}
    args['init_controllers'] = task_needs_gazebo or world_type_needs_controllers

    n = args.pop('name')
    if not n:
        n = []
        for k, v in sorted(args.items()):
            if (v != parser.get_default(k)) and (k not in ['env', 'seed', 'load_best_defaults', 'evaluation_only',
                                                           'vis_env', 'restore_model_path', 'eval_tasks', 'eval_worlds',
                                                           'total_steps', 'perform_collision_check',
                                                           'init_controllers', ]):
                n.append(str(v) if (type(v) == str) else f'{k}:{v}')
        n = '_'.join(n)
    run_name = '_'.join([j for j in [args['env'], n] if j])

    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['logpath'] = f'{config_path}/logs/'
    os.makedirs(args['logpath'], exist_ok=True)

    if args['seed'] == -1:
        args['seed'] = random.randint(10, 1000)
    set_seed(args['seed'])

    print(f"Log path: {args['logpath']}")
    return run_name, args


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    # need a node for visualisation
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    run_name, config = parse_args(main_path)
    logpath = f"{config['logpath']}{run_name}/"

    # create envs
    env = create_env(config, task=config['task'], node_handle="train_env", wrap_in_dummy_vec=True, flatten_obs=True)
    eval_config = dict(config).copy()
    eval_config["transition_noise_base"] = 0.0
    eval_config["ik_fail_thresh"] = config['ik_fail_thresh_eval']
    eval_config["node_handle"] = "eval_env"
    time.sleep(1)
    eval_env = create_env(eval_config, task=eval_config["task"], node_handle="eval_env", wrap_in_dummy_vec=True,
                          flatten_obs=True)

    agent = MobileRLLearner(env,
                            lr=config['lr'],
                            iters=config['total_steps'],
                            batch_size=config['batch_size'],
                            seed=config['seed'],
                            buffer_size=config['buffer_size'],
                            learning_starts=config['rnd_steps'],
                            tau=config['tau'],
                            gamma=config['gamma'],
                            explore_noise=config['explore_noise'],
                            explore_noise_type=config['explore_noise_type'],
                            nr_evaluations=config['nr_evaluations'],
                            evaluation_frequency=config['evaluation_frequency'],
                            checkpoint_after_iter=config['checkpoint_after_iter'],
                            restore_model_path=config['restore_model_path'],
                            checkpoint_load_iter=config['checkpoint_load_iter'],
                            temp_path=logpath,
                            device=config['device'],
                            ent_coef=config['ent_coef'])

    # train
    if not config['evaluation_only']:
        agent.fit(env, val_env=eval_env)
    else:
        # set up logger which otherwise is automatically handled by stable-baselines
        configure_logger(0, logpath, 'SAC', agent.stable_bl_agent.num_timesteps)

    # evaluate
    world_types = ["world"] if (config['world_type'] == "world") else config['eval_worlds']

    for world_type in world_types:
        for task in config['eval_tasks']:
            evaluate_on_task(config, eval_env_config=eval_config, agent=agent, task=task, world_type=world_type)

    rospy.signal_shutdown("We are done")


if __name__ == '__main__':
    main()
