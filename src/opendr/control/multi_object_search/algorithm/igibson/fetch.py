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


import gym
import numpy as np
from igibson.robots.fetch_robot import Fetch
from igibson.robots.robot_locomotor import LocomotorRobot


class Fetch_DD(Fetch):
    def __init__(self, config):
        self.config = config
        self.wheel_dim = 2
        self.wheel_axle_half = 0.186
        self.wheel_radius = 0.0613  # radius of the wheels
        self.linear_velocity = config.get("linear_velocity", 0.2)
        self.angular_velocity = config.get("angular_velocity", 0.1)

        LocomotorRobot.__init__(
            self,
            "fetch/fetch.urdf",
            action_dim=self.wheel_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="differential_drive",
            self_collision=config.get("self_collision", True))

    def set_up_continuous_action_space(self):
        self.action_high = np.zeros(self.wheel_dim)
        self.action_high[0] = self.linear_velocity
        self.action_high[1] = self.angular_velocity
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32)

    def apply_robot_action(self, action):

        lin_vel, ang_vel = action

        if not hasattr(self, "wheel_axle_half") or not hasattr(self, "wheel_radius"):
            raise Exception(
                "Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified."
            )
        left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
        right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius

        self.ordered_joints[1].set_motor_velocity(
            self.velocity_coef * left_wheel_ang_vel)  # *self.ordered_joints[1].max_velocity)
        self.ordered_joints[0].set_motor_velocity(
            self.velocity_coef * right_wheel_ang_vel)  # *self.ordered_joints[0].max_velocity)
