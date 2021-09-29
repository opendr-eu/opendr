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

import os

from pybindings import RobotObs, EEObs, LinearPlanner, GMMPlanner

MIN_PLANNER_VELOCITY = 0.001
MAX_PLANNER_VELOCITY = 0.1
# also defined in robot_env.cpp!
TIME_STEP_TRAIN = 0.1


class EEPlanner:
    def __init__(self,
                 gripper_goal_tip,
                 gripper_goal_wrist,
                 head_start,
                 map):
        self.gripper_goal_tip = gripper_goal_tip
        self.gripper_goal_wrist = gripper_goal_wrist
        self._head_start = head_start
        self._map = map

    def reset(self,
              robot_obs: RobotObs,
              slow_down_factor: float,
              is_analytic_env: bool,
              success_thres_dist: float,
              success_thres_rot: float) -> EEObs:
        raise NotImplementedError()

    def step(self, robot_obs: RobotObs, learned_vel_norm: float) -> EEObs:
        raise NotImplementedError()

    def generate_obs_step(self, robot_state: RobotObs) -> EEObs:
        raise NotImplementedError()


class LinearPlannerWrapper(EEPlanner):
    def __init__(self,
                 gripper_goal_tip,
                 gripper_goal_wrist,
                 head_start,
                 map):
        super(LinearPlannerWrapper, self).__init__(gripper_goal_tip,
                                                   gripper_goal_wrist,
                                                   head_start,
                                                   map)
        self._planner = None

    def reset(self,
              robot_obs: RobotObs,
              slow_down_factor: float,
              is_analytic_env: bool,
              success_thres_dist: float,
              success_thres_rot: float) -> EEObs:
        self._planner = LinearPlanner(self.gripper_goal_wrist,
                                      robot_obs.gripper_tf,
                                      [0, 0, 0, 0, 0, 0, 1],
                                      robot_obs.base_tf,
                                      success_thres_dist,
                                      success_thres_rot,
                                      MIN_PLANNER_VELOCITY,
                                      MAX_PLANNER_VELOCITY,
                                      slow_down_factor,
                                      self._head_start,
                                      TIME_STEP_TRAIN,
                                      is_analytic_env)
        return self.generate_obs_step(robot_obs)

    def step(self, robot_obs: RobotObs, learned_vel_norm: float) -> EEObs:
        return self._planner.step(robot_obs, learned_vel_norm)

    def generate_obs_step(self, robot_state: RobotObs) -> EEObs:
        return self._planner.generate_obs_step(robot_state)


class GMMPlannerWrapper(EEPlanner):
    def __init__(self,
                 gripper_goal_tip,
                 gripper_goal_wrist,
                 head_start,
                 map,
                 gmm_model_path: str,
                 robot_config):
        super(GMMPlannerWrapper, self).__init__(gripper_goal_tip,
                                                gripper_goal_wrist,
                                                head_start,
                                                map)
        self._planner = None
        assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"
        self._gmm_model_path = gmm_model_path
        self._robot_config = robot_config

    def reset(self,
              robot_obs: RobotObs,
              slow_down_factor: float,
              is_analytic_env: bool,
              success_thres_dist,
              success_thres_rot) -> EEObs:
        # NOTE: planners either take in the goal for the tip or the wrist, but always output plans for the wrist!
        self._planner = GMMPlanner(self.gripper_goal_wrist,
                                   robot_obs.gripper_tf,
                                   [0, 0, 0, 0, 0, 0, 1],
                                   robot_obs.base_tf,
                                   success_thres_dist,
                                   success_thres_rot,
                                   MIN_PLANNER_VELOCITY,
                                   MAX_PLANNER_VELOCITY,
                                   slow_down_factor,
                                   self._head_start,
                                   TIME_STEP_TRAIN,
                                   is_analytic_env,
                                   self._robot_config["tip_to_gripper_offset"],
                                   self._robot_config["gripper_to_base_rot_offset"],
                                   str(self._gmm_model_path),
                                   self._robot_config["gmm_base_offset"])
        return self.generate_obs_step(robot_obs)

    def step(self, robot_obs: RobotObs, learned_vel_norm: float) -> EEObs:
        return self._planner.step(robot_obs, learned_vel_norm)

    def generate_obs_step(self, robot_state: RobotObs) -> EEObs:
        return self._planner.generate_obs_step(robot_state)
