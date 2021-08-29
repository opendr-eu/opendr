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

import math
import numpy as np
import random
from collections import namedtuple
from enum import IntEnum
from gym import Wrapper
from typing import Tuple

from opendr.control.mobile_manipulation.mobileRL.envs.eeplanner import LinearPlannerWrapper
from opendr.control.mobile_manipulation.mobileRL.envs.map import Map, EmptyMap
from opendr.control.mobile_manipulation.mobileRL.envs.mobile_manipulation_env import MobileManipulationEnv


class GripperActions(IntEnum):
    NONE = 0
    OPEN = 1
    GRASP = 2


TaskGoal = namedtuple("TaskGoal", "gripper_goal_tip end_action success_thres_dist success_thres_rot head_start ee_fn")


def sample_circle_goal(goal_dist_rng: Tuple[float, float], goal_height_rng: Tuple[float, float]) -> list:
    goal_dist = random.uniform(goal_dist_rng[0], goal_dist_rng[1])
    goal_orientation = random.uniform(0.0, math.pi)

    # assuming currently at the origin!
    x = goal_dist * math.cos(goal_orientation)
    y = random.choice([-1, 1]) * goal_dist * math.sin(goal_orientation)
    z = random.uniform(goal_height_rng[0], goal_height_rng[1])

    RPY_goal = np.random.uniform(0, 2 * np.pi, 3)
    return [x, y, z] + RPY_goal.tolist()


class BaseTask(Wrapper):
    @staticmethod
    def taskname() -> str:
        raise NotImplementedError()

    @property
    def loggingname(self):
        return self.taskname() \
               + (f"_hs{self._default_head_start}" if self._default_head_start else "") \
               + (f"_{self.get_world()}" if self.get_world() != "sim" else "")

    @staticmethod
    def requires_simulator() -> bool:
        """
        Whether this task cannot be run in the analytical environment alone and needs e.g. the Gazebo simulator
        available (e.g. to spawn objects, deduct goals, ...)
        """
        raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __init__(self,
                 env: MobileManipulationEnv,
                 initial_joint_distribution: str,
                 map: Map,
                 default_head_start: float,
                 success_thres_dist: float = 0.025,
                 success_thres_rot: float = 0.05,
                 close_gripper_at_start: bool = True
                 ):
        super(BaseTask, self).__init__(env=env)
        self._initial_joint_distribution = initial_joint_distribution
        self.env.set_map(map)
        self._default_head_start = default_head_start
        self._success_thres_dist = success_thres_dist
        self._success_thres_rot = success_thres_rot
        self._close_gripper_at_start = close_gripper_at_start

    def draw_goal(self) -> TaskGoal:
        raise NotImplementedError()

    def reset(self, task_goal: TaskGoal = None):
        if task_goal is None:
            task_goal = self.draw_goal()
        ee_planner = task_goal.ee_fn(gripper_goal_tip=task_goal.gripper_goal_tip,
                                     gripper_goal_wrist=self.env.tip_to_gripper_tf(task_goal.gripper_goal_tip),
                                     head_start=task_goal.head_start,
                                     map=self.env.map)
        return self.env.reset(initial_joint_distribution=self._initial_joint_distribution,
                              success_thres_dist=task_goal.success_thres_dist,
                              success_thres_rot=task_goal.success_thres_rot,
                              ee_planner=ee_planner,
                              close_gripper=self._close_gripper_at_start)

    def step(self, action):
        return self.env.step(action=action)

    def clear(self):
        self.env.close()


class RndStartRndGoalsTask(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "rndStartRndGoal"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: MobileManipulationEnv, default_head_start, goal_dist_rng=(1, 5), goal_height_rng=None):
        map = EmptyMap(map_frame_rviz=env.robot_config['frame_id'] if env.get_world() == 'sim' else 'map',
                       inflation_radius=env.get_inflation_radius())
        super(RndStartRndGoalsTask, self).__init__(env=env, initial_joint_distribution='rnd',
                                                   map=map,
                                                   default_head_start=default_head_start)

        if goal_height_rng is None:
            goal_height_rng = (env.robot_config["z_min"], env.robot_config["z_max"])
        assert len(goal_dist_rng) == len(goal_height_rng) == 2
        self._goal_dist_rng = goal_dist_rng
        self._goal_height_rng = goal_height_rng

    def draw_goal(self):
        # assumes we are currently at the origin!
        gripper_goal_wrist = sample_circle_goal(self._goal_dist_rng, self._goal_height_rng)
        # first transform to tip here so that we are sampling in the wrist frame: easier to prevent impossible goals
        gripper_goal_tip = self.env.gripper_to_tip_tf(gripper_goal_wrist)

        return TaskGoal(gripper_goal_tip=gripper_goal_tip,
                        end_action=GripperActions.NONE,
                        success_thres_dist=self._success_thres_dist,
                        success_thres_rot=self._success_thres_rot,
                        head_start=self._default_head_start,
                        ee_fn=LinearPlannerWrapper)


class RestrictedWsTask(RndStartRndGoalsTask):
    @staticmethod
    def taskname() -> str:
        return "restrictedWs"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: MobileManipulationEnv, default_head_start):
        super(RestrictedWsTask, self).__init__(env=env,
                                               default_head_start=default_head_start,
                                               goal_height_rng=(env.robot_config["restricted_ws_z_min"],
                                                                env.robot_config["restricted_ws_z_max"]))
