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

import copy
import numpy as np
import rospy
from gym import Env
from pathlib import Path
from pybindings import RobotObs, RobotPR2, RobotTiago  # , RobotHSR
from typing import Tuple


class ActionRanges:
    @classmethod
    def get_ranges(cls, env_name: str, learn_vel_norm: bool):
        ks = []
        if learn_vel_norm:
            ks.append('vel_norm')

        if env_name == 'tiago':
            ks += ['tiago_base_vel', 'tiago_base_angle']
        else:
            ks += ['base_rot', 'base_x', 'base_y']

        n = len(ks)
        min_actions = n * [-1]
        max_actions = n * [1]

        return ks, np.array(min_actions), np.array(max_actions)


def rospy_maybe_initialise(node_name: str, anonymous: bool):
    """Check if there is already an initialised rospy node and initialise it if not"""
    try:
        rospy.get_time()
    except:
        rospy.init_node(node_name, anonymous=anonymous)


def unscale_action(scaled_action, low: float, high: float):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param scaled_action: Action to un-scale
    """
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


class RobotEnv(Env):
    def __init__(self,
                 env: str,
                 node_handle_name: str,
                 penalty_scaling: float,
                 time_step_world: float,
                 seed: int,
                 strategy: str,
                 world_type: str,
                 init_controllers: bool,
                 vis_env: bool,
                 transition_noise_base: float,
                 perform_collision_check: bool,
                 hsr_ik_slack_dist: float = None,
                 hsr_ik_slack_rot_dist: float = None,
                 hsr_sol_dist_reward: bool = None):
        conf_path = Path(__file__).parent.parent.parent / "robots_world" / env / "robot_config.yaml"
        assert conf_path.exists(), conf_path
        args = [seed,
                strategy,
                world_type,
                init_controllers,
                penalty_scaling,
                time_step_world,
                perform_collision_check,
                node_handle_name,
                vis_env,
                str(conf_path.absolute())]
        if env == 'pr2':
            self._env = RobotPR2(*args)
        elif env == 'tiago':
            self._env = RobotTiago(*args)
        # elif env == 'hsr':
        #     assert hsr_ik_slack_dist is not None
        #     assert hsr_ik_slack_rot_dist is not None
        #     assert hsr_sol_dist_reward is not None
        #     self._env = RobotHSR(*args, hsr_ik_slack_dist, hsr_ik_slack_rot_dist, hsr_sol_dist_reward)
        else:
            raise ValueError('Unknown env')

        if vis_env:
            rospy_maybe_initialise(node_handle_name + '_py', anonymous=True)

        self._strategy = strategy
        self.env_name = env
        self._transition_noise_base = transition_noise_base
        self.reset(initial_base_pose=[0, 0, 0, 0, 0, 0, 1], initial_joint_distribution="rnd", success_thres_dist=0.05,
                   success_thres_rot=0.05)

        self.robot_config = self._get_robot_config()
        self.vis_env = vis_env

        # NOTE: atm only supporting rectangles, while HSR, Tiago are circles
        self.robot_base_size = (self.robot_config["robot_base_size_meters_y"],
                                self.robot_config["robot_base_size_meters_x"])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

    def normalize_reward(self, reward):
        # needed for stable_baselines replay buffer class
        return reward

    def reset(self,
              initial_base_pose: list,
              initial_joint_distribution: str,
              success_thres_dist: float,
              success_thres_rot: float,
              close_gripper: bool = False) -> RobotObs:
        assert initial_joint_distribution in ["fixed", "restricted_ws", "rnd"], initial_joint_distribution
        if initial_base_pose is None:
            initial_base_pose = [0, 0, 0] + [0, 0, 0, 1]

        self._nr_kin_failures = 0
        self.trajectory = []

        return self._env.reset(initial_base_pose,
                               initial_joint_distribution,
                               close_gripper,
                               success_thres_dist,
                               success_thres_rot)

    def step(self, base_actions, ee_velocities_world) -> Tuple[RobotObs, dict]:
        robot_obs = self._env.step(base_actions, ee_velocities_world, self._transition_noise_base)
        self._nr_kin_failures += robot_obs.ik_fail
        self.trajectory.append([robot_obs.base_tf, robot_obs.gripper_tf])
        info = {'nr_kin_failures': self._nr_kin_failures}
        return robot_obs, info

    def close(self):
        pass

    def open_gripper(self, position: float = 0.08, wait_for_result: bool = True):
        if self.is_analytical_world():
            return True
        else:
            return self._env.open_gripper(position, wait_for_result)

    def close_gripper(self, position: float = 0.00, wait_for_result: bool = True):
        if self.is_analytical_world():
            return True
        else:
            return self._env.close_gripper(position, wait_for_result)

    def set_ik_slack(self, ik_slack_dist: float, ik_slack_rot_dist: float):
        assert self.env_name == 'hsr'
        self._env.set_ik_slack(ik_slack_dist, ik_slack_rot_dist)

    def _get_robot_config(self) -> dict:
        scalar_conf, str_conf = copy.deepcopy(self._env.get_robo_config())
        for k, v in scalar_conf.items():
            if len(v) == 1:
                scalar_conf[k] = v[0]
        scalar_conf.update(str_conf)
        return scalar_conf

    def publish_marker(self, marker_tf, marker_id: int, namespace: str, color: str, alpha: float,
                       geometry: str = "arrow", marker_scale=(0.1, 0.025, 0.025)):
        self._env.publish_marker(marker_tf, marker_id, namespace, color, alpha, geometry, marker_scale)
