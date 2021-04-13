import argparse
import collections
import copy
import os
import random
import shutil
import sys
import time
from typing import List

import numpy as np
import rospy
import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

from modulation import __version__
from modulation.envs import ALL_TASKS
from modulation.envs.combined_env import CombinedEnv
from modulation.envs.robotenv import RobotEnv


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(config_path, using_wandb: bool = False):
    all_tasks = ALL_TASKS.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_best_defaults', type=str2bool, nargs='?', const=True, default=False, help="Replace default values with those from configs/best_defaults.yaml.")
    parser.add_argument('--seed', type=int, default=-1, help="Set to a value >= to use deterministic seed")
    parser.add_argument('--rnd_steps', type=int, default=0, help='Number of random actions to record before starting with rl, subtracted from total_steps')
    parser.add_argument('--total_steps', type=int, default=1_000_000, help='Total number of action/observation steps to take over all episodes')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    #################################################
    # ALGORITHMS
    #################################################
    parser.add_argument('--algo', type=str.lower, default='sac', choices=['td3', 'sac'])
    parser.add_argument('--gamma', type=float, default=0.99, help='discount')
    parser.add_argument('--lr_start', type=float, default=1e-5)
    parser.add_argument('--lr_end', type=float, default=1e-6, help="Final / min learning rate. -1 to not decay")
    # parser.add_argument('--lr_gamma', type=float, default=0.999, help='adam decay')
    parser.add_argument('--tau', type=float, default=0.001, help='target value moving average speed')
    parser.add_argument('--explore_noise_type', type=str, default='normal', choices=['normal', 'OU', ''], help='Type of exploration noise')
    parser.add_argument('--explore_noise', type=float, default=0.0, help='')
    #################################################
    # TD3
    #################################################
    parser.add_argument('--policy_noise', type=float, default=0.1, help='noise added to target policy in critic update')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='range to clip target policy noise')
    parser.add_argument('--policy_frequency', type=int, default=2, help='Frequency with which the policy will be updated')
    #################################################
    # SAC
    #################################################
    parser.add_argument('--use_sde', type=str2bool, nargs='?', const=True, default=False, help="use sde exploration instead of action noise. Automatically sets explore_noise_type to None")
    parser.add_argument('--ent_coef', default="auto", help="Entropy coefficient. 'auto' to learn it.")
    #################################################
    # Env
    #################################################
    parser.add_argument('--env', type=str.lower, default='pr2', choices=['pr2', 'tiago', 'hsr'], help='')
    parser.add_argument('--use_map_obs', type=str2bool, nargs='?', const=True, default=True, help='Observe a local obstacle map')
    parser.add_argument('--task', type=str.lower, default='rndstartrndgoal', choices=all_tasks, help='Train on a specific task env. Might override some other choices.')
    parser.add_argument('--obstacle_config', type=str.lower, default='none', choices=['none', 'inpath'], help='Obstacle configuration for ObstacleConfigMap. Ignored for all other tasks')
    parser.add_argument('--time_step', type=float, default=0.02, help='Time steps at which the RL agent makes decisions during actual execution. NOTE: time_step for training is hardcoded in robot_env.cpp.')
    parser.add_argument('--slow_down_real_exec', type=float, default=1.0, help='How much to slow down the planned gripper trajectories during real execun')
    parser.add_argument('--world_type', type=str, default="sim", choices=["sim", "gazebo", "world"], help="What kind of movement execution and where to get updated values from. Sim: analytical environemt, don't call controllers, gazebo: gazebo simulator, world: real world")
    parser.add_argument('--stack_k_obs', type=int, default=1, help='number of (past) obs to stack and return as current obs. 1 to just return current obs')
    parser.add_argument('--strategy', type=str.lower, default="dirvel", choices=["relvelm", "relveld", "dirvel", "modulate_ellipse", "unmodulated"], help='What velocities to learn: modulate, velocity relative to the gripper velocity, direct base velocity')
    parser.add_argument('--ik_fail_thresh', type=int, default=20, help='number of failures after which on it is considered as failed (i.e. failed: failures > ik_fail_thresh)')
    parser.add_argument('--ik_fail_thresh_eval', type=int, default=100, help='different eval threshold to make comparable across settings and investigate if it can recover from failures')
    parser.add_argument('--penalty_scaling', type=float, default=0.01, help='by how much to scale the penalties to incentivise minimal modulation')
    parser.add_argument('--learn_vel_norm', type=float, default=-1, help="Learn the norm of the next EE-motion. Value is the factor weighting the loss for this. -1 to not learn it.")
    parser.add_argument('--perform_collision_check', type=str2bool, nargs='?', const=True, default=True, help='Use the planning scen to perform collision checks (both with environment and self collisions)')
    parser.add_argument('--vis_env', type=str2bool, nargs='?', const=True, default=False, help='Whether to publish markers to rviz')
    # parser.add_argument('--transition_noise_ee', type=float, default=0.0, help='Std of Gaussian noise applied to the next gripper transform during training')
    parser.add_argument('--transition_noise_base', type=float, default=0.0, help='Std of Gaussian noise applied to the next base transform during training')
    parser.add_argument('--head_start', type=float, default=0.0, help='Seconds to wait before starting the EE-motion (allowing the base to position itself)')
    #################################################
    # HSR
    #################################################
    parser.add_argument('--hsr_ik_slack_dist', type=float, default=0.1, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_ik_slack_rot_dist', type=float, default=0.05, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_sol_dist_reward', type=str2bool, default=False, help='Penalise distance to perfect ik solution')
    #################################################
    # Eval
    #################################################
    parser.add_argument('--nr_evaluations', type=int, default=50, help='Nr of runs for the evaluation')
    parser.add_argument('--evaluation_frequency', type=int, default=20000, help='In nr of steps')
    parser.add_argument('--evaluation_only', type=str2bool, nargs='?', const=True, default=False, help='If True only model will be loaded and evaluated no training')
    parser.add_argument('--eval_execs', nargs='+', default=['sim'], choices=['sim', 'gazebo', 'world'], help='Eval execs to run')
    parser.add_argument('--eval_tasks', nargs='+', default=['rndstartrndgoal', 'houseexpo', 'simpleobstacle', 'restrictedws'], choices=all_tasks, help='Eval tasks to run')
    # #################################################
    # wandbstuff
    #################################################
    parser.add_argument('--resume_id', type=str, default=None, help='wandb id to resume')
    parser.add_argument('--resume_model_name', type=str, default='last_model.zip', help='If specifying a resume_id, which model to restore')
    parser.add_argument('--restore_model', type=str2bool, nargs='?', const=True, default=False, help='Restore the model and config saved in /scripts/model_checkpoints/${env}/.')
    parser.add_argument('--name', type=str, default="", help='wandb display name for this run')
    parser.add_argument('--name_suffix', type=str, default="", help='suffix for the wandb name')
    parser.add_argument('--tags', type=str, nargs='+', default=[], help='wandb tags')
    parser.add_argument('--group', type=str, default=None, help='wandb group')
    parser.add_argument('--use_name_as_group', type=str2bool, nargs='?', const=True, default=True, help='use the name as group')
    parser.add_argument('--project_name', type=str, default='mobile_rl', help='wandb project name')
    parser.add_argument('-d', '--dry_run', type=str2bool, nargs='?', const=True, default=False, help='whether not to log this run to wandb')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='log gradients to wandb, potentially extra verbosity')

    args = parser.parse_args()
    args = vars(args)

    # user-specified command-line arguments
    cl_args = [k.replace('-', '').split('=')[0] for k in sys.argv]

    if args.pop('load_best_defaults'):
        with open(config_path / 'configs' / 'best_defaults.yaml') as f:
            best_defaults = yaml.safe_load(f)
        # replace with best_default value unless something else was specified through command line
        for k, v in best_defaults[args['env']].items():
            if k not in cl_args:
                args[k] = v

    # consistency checks for certain arguments
    if args['strategy'] in ['modulate_ellipse', 'unmodulated']:
        assert args['algo'] == 'unmodulated'
        assert args['evaluation_only']
        if args['strategy'] == 'modulate_ellipse':
            assert args['env'] == 'pr2', args['env']
    if args['algo'] == 'unmodulated':
        assert args['strategy'] in ['modulate_ellipse', 'unmodulated']
    if args['use_sde']:
        print("Using sde, setting explore_noise_type to None")
        args['explore_noise_type'] = None
    if args['resume_id'] or args['restore_model']:
        assert args['evaluation_only'], "Continuing to train not supported atm (replay buffer doesn't get saved)"
    if args['env'] == 'hsr' and args['perform_collision_check']:
        print("SETTING perform_collision_check TO FALSE FOR HSR (RISK OF CRASHING GAZEBO)")
        args['perform_collision_check'] = False
    if args['env'] == 'hsr':
        assert not args['perform_collision_check'], "Collisions seem to potentially crash due to some unsupported geometries"
    if args['evaluation_only']:
        if not (args['resume_id'] or args['restore_model'] or (args['algo'] == 'unmodulated')):
            print("Evaluation only but no model to load specified! Evaluating a randomly initialised agent.")

    # do we need to initialise controllers for the specified tasks?
    tasks_that_need_controllers = [k for k, v in ALL_TASKS.items() if v.requires_simulator()]
    task_needs_gazebo = len(set([args['task']] + args['eval_tasks']).intersection(set(tasks_that_need_controllers))) > 0
    world_type_needs_controllers = set([args['world_type']] + args['eval_execs']) != {'sim'}
    args['init_controllers'] = task_needs_gazebo or world_type_needs_controllers

    n = args.pop('name')
    group = args.pop('group')
    use_name_as_group = args.pop('use_name_as_group')
    if not n:
        n = []
        for k, v in sorted(args.items()):
            if (v != parser.get_default(k)) and (k not in ['env', 'seed', 'load_best_defaults', 'name_suffix',
                                                           'evaluation_only', 'vis_env',
                                                           'resume_id', 'eval_tasks', 'eval_execs', 'total_steps',
                                                           'perform_collision_check', 'init_controllers',
                                                           'num_workers', 'num_cpus_per_worker', 'num_envs_per_worker',
                                                           'num_gpus', 'num_gpus_per_worker', 'ray_verbosity', ]):
                n.append(str(v) if (type(v) == str) else f'{k}:{v}')
        n = '_'.join(n)
    run_name = '_'.join([j for j in [args['env'], n, args.pop('name_suffix')] if j])

    if use_name_as_group:
        assert not group, "Don't specify a group and use_name_as_group"
        rname = run_name[:99] if len(run_name) > 99 else run_name
        group = rname + f'_v{__version__}'

    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['logpath'] = f'{config_path}/logs'
    os.makedirs(args['logpath'], exist_ok=True)
    args['version'] = __version__

    if not using_wandb:
        if args['seed'] == -1:
            args['seed'] = random.randint(10, 1000)
        set_seed(args['seed'])

    print(f"Log path: {args['logpath']}")

    return run_name, group, args, cl_args


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_config_wandb(config_path, tags: List = None, sync_tensorboard=False, allow_init: bool = True,
                       no_ckpt_endig: bool = False):
    run_name, group, args, cl_args = parse_args(config_path, using_wandb=True)

    if args['nstep'] != 1:
        raise NotImplementedError("Currently need to use the nstep branch of the stableBL repo to have this supported")
        assert args['nstep'] > 0
        assert args['algo'] == 'td3', "Not correctly implemented for SAC yet"

    if tags is None:
        tags = []

    if args['dry_run']:
        os.environ['WANDB_MODE'] = 'dryrun'

    if no_ckpt_endig:
        args['resume_model_name'] = args['resume_model_name'].replace('.zip', '')

    common_args = {'project': args.pop('project_name'),
                   'dir': args['logpath'],
                   'tags': [f'v{__version__}'] + tags + args.pop('tags'),
                   'sync_tensorboard': sync_tensorboard,
                   'group': group}
    if args['resume_id']:
        assert not args['dry_run']
        run = wandb.init(id=args['resume_id'],
                         resume=args['resume_id'],
                         **common_args)
    elif args['restore_model']:
        ckpt_path = config_path / 'model_checkpoints' / args['env']
        print(f"RESTORING MODEL FOR {args['env']} from {ckpt_path}")

        with open(ckpt_path / 'config.yaml') as f:
            raw_params = yaml.safe_load(f)
        params = {k: v['value'] for k, v in raw_params.items() if k not in ['_wandb', 'wandb_version']}

        params['model_file'] = ckpt_path / args['resume_model_name']
        params['restore_model'] = True
        params['resume_id'] = None

        run = wandb.init(config=params, **common_args)
        if args['evaluation_only']:
            wandb.config.update({"evaluation_only": True}, allow_val_change=True)
    else:
        if allow_init:
            # delete all past wandb runs. Mostly relevant for running sweeps in docker which might fill up the space o/w
            delete_dir(os.path.join(common_args['dir'], 'wandb'))
            run = wandb.init(config=args, name=run_name, **common_args)
        else:
            raise ValueError("Not allowed to initialise a new run. Sepcify restore_model=True or a resume_id")

    if args['resume_id'] or args['restore_model']:
        # update an alternative dict placeholder so we don't change the logged values which it was trained with
        config = DotDict(copy.deepcopy(dict(wandb.config)))

        for k, v in args.items():
            # allow to override loaded config with command line args
            if k in cl_args:
                config[k] = v
            # backwards compatibility if a config value didn't exist before
            if k not in wandb.config.keys():
                print(f"Key {k} not found in config. Setting to {v}")
                config[k] = args[k]
        # always update these values
        for k in ['init_controllers', 'device', 'num_workers', 'num_cpus_per_worker', 'num_envs_per_worker', 'num_gpus',
                  'num_gpus_per_worker', 'logpath']:
            config[k] = args[k]
    else:
        config = wandb.config

    # Set seeds
    # NOTE: if changing the args wandb will not get the change in sweeps as they don't work over the command line!!!
    if config.seed == -1:
        wandb.config.update({"seed": random.randint(10, 1000)}, allow_val_change=True)
        config['seed'] = wandb.config.seed

    set_seed(config.seed)

    return run, config


class DotDict(dict):
    """
    Source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def delete_dir(dirname: str):
    try:
        print(f"Deleting dir {dirname}")
        shutil.rmtree(dirname)
    except Exception as e:
        print(f"Failed to delete dir {dirname}: {e}")


def rpy_to_quiver_uvw(roll, pitch, yaw):
    U = np.cos(yaw) * np.cos(pitch)
    V = np.sin(yaw) * np.cos(pitch)
    W = np.sin(pitch)
    return U, V, W


def wrap_in_task(env, task: str, default_head_start: float, wrap_in_dummy_vec: bool, **env_kwargs):
    if isinstance(env, DummyVecEnv):
        combined_env = env.envs[0].unwrapped
    elif isinstance(env, CombinedEnv):
        combined_env = env
    else:
        combined_env = env.unwrapped
    assert isinstance(combined_env, CombinedEnv), combined_env

    task_fn = ALL_TASKS[task.lower()]
    task_env = task_fn(combined_env, default_head_start, **env_kwargs)

    if task_env.requires_simulator():
        assert env._robot.get_init_controllers(), "We need gazebo to spawn objects etc"

    if wrap_in_dummy_vec:
        task_env = DummyVecEnv([lambda: task_env])
    return task_env


def create_env(config,
               task: str,
               node_handle: str,
               eval: bool,
               wrap_in_dummy_vec: bool = False,
               flatten_obs: bool = False) -> CombinedEnv:
    print(f"Creating {config['env']}")

    robot_env = RobotEnv(env=config["env"],
                         node_handle_name=node_handle,
                         penalty_scaling=config["penalty_scaling"],
                         time_step_world=config["time_step"],
                         seed=config["seed"],
                         strategy=config["strategy"],
                         world_type=config["world_type"],
                         init_controllers=config["init_controllers"],
                         perform_collision_check=config["perform_collision_check"],
                         vis_env=config["vis_env"],
                         transition_noise_base=config["transition_noise_base"],
                         hsr_ik_slack_dist=config["hsr_ik_slack_dist"],
                         hsr_ik_slack_rot_dist=config["hsr_ik_slack_rot_dist"],
                         hsr_sol_dist_reward=config["hsr_sol_dist_reward"])
    env = CombinedEnv(robot_env=robot_env,
                      ik_fail_thresh=config["ik_fail_thresh"],
                      learn_vel_norm=config["learn_vel_norm"],
                      slow_down_real_exec=config["slow_down_real_exec"],
                      use_map_obs=config["use_map_obs"],
                      flatten_obs=flatten_obs)
    if task == 'adversarial':
        env_kwargs = {'adversary_max_steps': config["adversary_max_steps"]}
        assert config["use_map_obs"]
    elif task == 'hadversarial':
        env_kwargs = {'adversary_max_steps': config["adversary_max_steps"],
                      'n_players': config["n_players"],
                      'player_gamma': config.get('player_gamma', None) or config.get('gamma')}
        assert config["use_map_obs"]
    elif task in ['picknplace', 'door', 'drawer']:
        env_kwargs = {'obstacle_configuration': config['obstacle_config']}
    elif task == 'houseexpo':
        env_kwargs = {'eval': eval}
    else:
        env_kwargs = {}
    return wrap_in_task(env=env, task=task, default_head_start=config["head_start"],
                        wrap_in_dummy_vec=wrap_in_dummy_vec, **env_kwargs)


def episode_is_success(nr_kin_fails: int, nr_collisions: int, goal_reached: bool) -> bool:
    return (nr_kin_fails == 0) and (nr_collisions == 0) and goal_reached


def env_creator(ray_env_config: dict, flatten_obs: bool = False):
    """Allows to construct a different eval env by defining 'task': eval_task in 'evaluation_config'"""
    time.sleep(random.uniform(0.0, 0.2))
    env = create_env(ray_env_config,
                     task=ray_env_config['task'],
                     node_handle=ray_env_config["node_handle"],
                     eval=ray_env_config['eval'],
                     flatten_obs=flatten_obs)
    return env
