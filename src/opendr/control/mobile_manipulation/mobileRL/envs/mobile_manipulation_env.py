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

import numpy as np
from gym import spaces, Env
from pybindings import RobotObs

from opendr.control.mobile_manipulation.mobileRL.envs.eeplanner import EEPlanner
from opendr.control.mobile_manipulation.mobileRL.envs.map import Map, DummyMap
from opendr.control.mobile_manipulation.mobileRL.envs.robotenv import RobotEnv, ActionRanges, unscale_action


class MobileManipulationEnv(Env):
    metadata = {'render.modes': []}

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __init__(self,
                 robot_env: RobotEnv,
                 ik_fail_thresh: int,
                 learn_vel_norm: bool,
                 slow_down_real_exec: float,
                 flatten_obs: bool = False):
        self._robot = robot_env
        self._ee_planner = None
        # each task wrapper should set a map at initialisation if needed
        # placeholder with no computational costs for when not needed
        self.map = DummyMap()

        self._ik_fail_thresh = ik_fail_thresh
        self._learn_vel_norm = learn_vel_norm
        self._slow_down_real_execution = slow_down_real_exec

        self._use_map_obs = False
        # Only so we can easily keep compatibility with stable-baselines which needs flat observations
        self._flatten_obs = flatten_obs
        self.robot_state_dim = self._robot.get_obs_dim()
        print(f"Detected robot state dim: {self.robot_state_dim}")

        self.action_names, self._min_actions, self._max_actions = ActionRanges.get_ranges(env_name=self._robot.env_name,
                                                                                          learn_vel_norm=(
                                                                                                  learn_vel_norm != -1))
        print(f"Actions to learn: {self.action_names}")
        self.action_dim = len(self._min_actions)

        # NOTE: env does all the action scaling according to _min_actions and _max_actions
        # and just expects actions in the range [-1, 1] from the agent
        self.action_space = spaces.Box(low=np.array(self.action_dim * [-1.0]),
                                       high=np.array(self.action_dim * [1.0]),
                                       shape=[self.action_dim])
        self.reward_range = (-10, 0)

    def get_inflation_radius(self) -> float:
        return max(self._robot.robot_base_size) / 2

    @property
    def observation_space(self):
        if self._flatten_obs:
            n = self.robot_state_dim
            if self._use_map_obs:
                n += np.prod(self.map.output_size)
            return spaces.Box(low=-100, high=100, shape=[n])
        else:
            s = spaces.Box(low=-100, high=100, shape=[self.robot_state_dim])
            if self._use_map_obs:
                s = spaces.Tuple([s,
                                  # map atm returns a 2D image w/o any channel
                                  spaces.Box(low=-0.1, high=1.1, shape=list(self.map.output_size) + [1],
                                             dtype=np.float)])
            return s

    def set_map(self, map: Map):
        self.map = map

    def set_ee_planner(self,
                       ee_planner: EEPlanner,
                       success_thres_dist: float,
                       success_thres_rot: float):
        is_analytical = self._robot.is_analytical_world()
        robot_obs = self._robot.get_robot_obs()

        self._ee_planner = ee_planner
        ee_obs_train_freq = self._ee_planner.reset(success_thres_dist=success_thres_dist,
                                                   success_thres_rot=success_thres_rot,
                                                   robot_obs=robot_obs,
                                                   slow_down_factor=self._slow_down_real_execution if not is_analytical else 1,
                                                   is_analytic_env=is_analytical)

        if self._robot.env_name == "hsr":
            self._robot.set_gripper_goal_wrist(ee_planner.gripper_goal_wrist)
        # returns a full obs, not just the new ee_obs to avoid having to stick it together correctly in other places
        return self._to_agent_obs(robot_obs, ee_obs_train_freq.ee_velocities_rel)

    def reset(self,
              initial_joint_distribution: str,
              success_thres_dist: float,
              success_thres_rot: float,
              ee_planner: EEPlanner,
              close_gripper: bool = False):
        # first clear map so robot cannot crash into objects when resetting, then respawn the scene
        self.map.clear()
        self._robot.reset(initial_base_pose=self.map.draw_initial_base_pose(
            ee_planner.gripper_goal_wrist if ee_planner is not None else None),
            initial_joint_distribution=initial_joint_distribution,
            success_thres_dist=success_thres_dist,
            success_thres_rot=success_thres_rot,
            close_gripper=close_gripper)

        self.map.map_reset()
        # sometimes desirable to call set_ee_planner manually after the map has reset to first deduct goals from the map
        if ee_planner is not None:
            return self.set_ee_planner(ee_planner=ee_planner,
                                       success_thres_dist=success_thres_dist,
                                       success_thres_rot=success_thres_rot)

    def step(self, action):
        reward = 0.0
        vel_norm, base_actions = self._convert_policy_to_env_actions(action)

        ee_obs = self._ee_planner.step(self._robot.get_robot_obs(), vel_norm)
        robot_obs, robot_info = self._robot.step(base_actions, ee_obs.ee_velocities_world)
        ee_obs_train_freq = self._ee_planner.generate_obs_step(robot_obs)
        robot_info['ee_done'] = ee_obs_train_freq.done

        reward += robot_obs.reward + ee_obs_train_freq.reward
        if (self._learn_vel_norm != -1):
            reward -= self._learn_vel_norm * (self._robot.robot_config['base_vel_rng'] - vel_norm)

        done = (robot_obs.done and ee_obs_train_freq.done) or (robot_info['nr_kin_failures'] >= self._ik_fail_thresh)
        return self._to_agent_obs(robot_obs, ee_obs_train_freq.ee_velocities_rel), reward, done, robot_info

    def _to_agent_obs(self, robot_obs: RobotObs, ee_velocities_rel, local_map=None) -> list:
        obs_vector = robot_obs.relative_gripper_tf \
                     + ee_velocities_rel \
                     + self._robot.world_to_relative_tf(self._ee_planner.gripper_goal_wrist) \
                     + robot_obs.joint_values
        if not self._use_map_obs:
            return obs_vector
        else:
            if local_map is None:
                local_map = self.map.get_local_map(robot_obs.base_tf, self._robot.get_world())
            if self._flatten_obs:
                obs_vector += local_map.flatten().tolist()
                return obs_vector
            else:
                local_map = local_map[:, :, np.newaxis]
                return obs_vector, local_map

    def _convert_policy_to_env_actions(self, actions):
        # NOTE: all actions are in range [-1, 1] at this point. Scale into target range either here or in the robot env.
        actions = list(actions)

        min_learned_vel_norm = 0.01

        if (self._learn_vel_norm != -1):
            # into range [min_vel_norm, robot_env.base_vel_rng]
            vel_norm = unscale_action(actions.pop(0), min_learned_vel_norm, self._robot.robot_config["base_vel_rng"])
        else:
            vel_norm = -1

        base_actions = actions
        return vel_norm, base_actions
