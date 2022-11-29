#!/usr/bin/env python
# Copyright 2020-2022 OpenDR European Project
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

import rclpy


class MyRobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.__gps = self.__robot.getDevice('gps1')
        self.__imu = self.__robot.getDevice('inertial_unit')

        rclpy.init(args=None)
        self.__node = rclpy.create_node('my_robot_driver')

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        roll, pitch, yaw = self.__imu.getRollPitchYaw()
        v1, v2, v3 = self.__gps.getValues()
