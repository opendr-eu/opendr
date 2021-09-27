# Copyright 2020-2021 OpenDR European Project
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

from stable_baselines3.common.vec_env import DummyVecEnv

from opendr.control.mobile_manipulation.mobileRL.envs import ALL_TASKS
from opendr.control.mobile_manipulation.mobileRL.envs.mobile_manipulation_env import MobileManipulationEnv
from opendr.control.mobile_manipulation.mobileRL.envs.robotenv import RobotEnv


def wrap_in_task(env, task: str, default_head_start: float, wrap_in_dummy_vec: bool, **env_kwargs):
    if isinstance(env, DummyVecEnv):
        combined_env = env.envs[0].unwrapped
    elif isinstance(env, MobileManipulationEnv):
        combined_env = env
    else:
        combined_env = env.unwrapped
    assert isinstance(combined_env, MobileManipulationEnv), combined_env

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
               wrap_in_dummy_vec: bool = False,
               flatten_obs: bool = False) -> MobileManipulationEnv:
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
                         hsr_ik_slack_dist=config.get("hsr_ik_slack_dist", None),
                         hsr_ik_slack_rot_dist=config.get("hsr_ik_slack_rot_dist", None),
                         hsr_sol_dist_reward=config.get("hsr_sol_dist_reward", None))
    env = MobileManipulationEnv(robot_env=robot_env,
                                ik_fail_thresh=config["ik_fail_thresh"],
                                learn_vel_norm=config["learn_vel_norm"],
                                slow_down_real_exec=config["slow_down_real_exec"],
                                flatten_obs=flatten_obs)
    if task in ['picknplace', 'door', 'drawer']:
        env_kwargs = {'obstacle_configuration': config.get('obstacle_config', 'none')}
    else:
        env_kwargs = {}
    return wrap_in_task(env=env, task=task, default_head_start=config["head_start"],
                        wrap_in_dummy_vec=wrap_in_dummy_vec, **env_kwargs)
