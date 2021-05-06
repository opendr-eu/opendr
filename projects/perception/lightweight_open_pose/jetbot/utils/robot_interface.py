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
import time


class PoseRobot:

    def __init__(self, robot='webots'):
        """
        Initializes the robot interface
        @param robot: if set to "webots" interfaces with the webots simulator. Set to "jetbot" if you are running the
            controller in a real robot.
        """

        self.robot_interface = robot
        if self.robot_interface == 'webots':
            from controller import Robot, Camera, Supervisor
            self.robot = Supervisor()
            self.timestep = int(self.robot.getBasicTimeStep())
            self.camera = Camera('camera')
            self.camera.enable(self.timestep)
            self.motor_left = self.robot.getDevice('wheel_left_joint')
            self.motor_right = self.robot.getDevice('wheel_right_joint')

            self.max_speed = self.motor_left.getMaxVelocity()
            self.motor_left.setVelocity(0.8)
            self.motor_right.setVelocity(0.8)

            self.left_motor_position_sensor = self.motor_left.getPositionSensor()
            self.left_motor_position_sensor.enable(self.timestep)
            self.right_motor_position_sensor = self.motor_right.getPositionSensor()
            self.right_motor_position_sensor.enable(self.timestep)

            self.motor_left.setPosition(self.left_motor_position_sensor.getValue())
            self.motor_right.setPosition(self.right_motor_position_sensor.getValue())

            # Angle to use for right translation
            self.translation_angle = 6
            self.rotation_angle = 3
        elif self.robot_interface == 'jetbot':
            from jetbot import Camera, Robot
            self.robot = Robot()
            self.robot.set_motors(0, 0)
            self.robot_speed = 0.3
            self.timestep = 0.2

            self.camera = Camera.instance(width=1200, height=675)
        else:
            assert False

    def kill_switch(self):
        """
        This function acts as a killswitch, it immediately stops the robot
        @return:
        @rtype:
        """
        if self.robot_interface == 'webots':
            self.motor_left.setPosition(self.left_motor_position_sensor.getValue())
            self.motor_right.setPosition(self.right_motor_position_sensor.getValue())
        elif self.robot_interface == 'jetbot':
            self.robot.stop()

    def step(self, steps=1):
        """
        Performs one simulation step. It just sleep for a predefined amount of time to allow for completing an action
        for real deployment. When in simulation, it allows for altering the state of the simulator by setting the
        human_state variable.

        @param steps: number of simulation steps to perform or number of timeslots to pass (when not in simulation)
        """
        if self.robot_interface == 'webots':
            return self.robot.step(self.timestep * steps)
        elif self.robot_interface == 'jetbot':
            return True
        else:
            assert False

    def translate(self, distance, visualizer_fn=None):
        """
        Translates the robot either forwards or backwards.
        @param distance: distance to be covered (either positive or negative). Please note that distance actually refers
        to timeslots for which the motors will operate at predefined speed.
        @type distance:
        @return:
        @rtype:
        """
        if distance > 0:
            self._translate_forward(distance, visualizer_fn)
        else:
            self._translate_backward(-distance, visualizer_fn)

    def rotate(self, rotation_dist, visualizer_fn=None):
        """
        Rotates a robot either to the right or to the left according to the distance defined in rotation
        @param rotation_dist: distance to be covered (either positive for left rotation or negative for right rotation)
        @param visualizer_fn: A function to call for visualizing the intermediate steps
        """
        if rotation_dist > 0:
            self._rotate_left(rotation_dist, visualizer_fn)
        else:
            self._rotate_right(-rotation_dist, visualizer_fn)

    def get_camera_data(self):
        """
        Returns a frame captured by the on-board camera of the robot
        @return:
        @rtype:
        """
        if self.robot_interface == 'webots':
            raw_img = self.camera.getImage()
            img = np.frombuffer(raw_img, np.uint8)
            img = np.float32(img.reshape((1080, 1920, 4))[:, :, :3])
            return img
        elif self.robot_interface == 'jetbot':
            raw_img = self.camera.value
            img = np.float32(raw_img)
            return img
        else:
            assert False

    def _rotate_right(self, rotation_dist=5, visualizer_fn=None):
        """
        Rotates the robot to right by defining the distance to be covered/time to be active by the left motor
        @param rotation_dist: distance to be covered
        """

        if self.robot_interface == 'webots':
            # Turn to the right
            left_target = self.left_motor_position_sensor.getValue() + rotation_dist
            self.motor_left.setPosition(left_target)
            while (np.abs(self.left_motor_position_sensor.getValue() - left_target) > 0.001):
                self.robot.step(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            if visualizer_fn:
                visualizer_fn(self.get_camera_data())
        elif self.robot_interface == 'jetbot':
            self.robot.set_motors(self.robot_speed, -self.robot_speed)
            for i in range(int(rotation_dist / self.timestep)):
                time.sleep(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            self.robot.stop()
        else:
            assert False

    def _rotate_left(self, rotation_dist=5, visualizer_fn=None):
        """
        Rotates the robot to left by defining the distance to be covered by the right motor
        @param rotation_dist: distance to be covered
        """

        if self.robot_interface == 'webots':
            # Turn to the left
            right_target = self.right_motor_position_sensor.getValue() + rotation_dist
            self.motor_right.setPosition(right_target)
            while (np.abs(self.right_motor_position_sensor.getValue() - right_target) > 0.001):
                self.robot.step(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
        elif self.robot_interface == 'jetbot':
            self.robot.set_motors(-self.robot_speed, self.robot_speed)
            for i in range(int(rotation_dist / self.timestep)):
                time.sleep(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            self.robot.stop()
        else:
            assert False

    def _translate_forward(self, duration=2, visualizer_fn=None):
        """
        Moves the robot forwards for duration timeslots
        @param duration: number of timeslots for which the motors will operate
        """

        if self.robot_interface == 'webots':
            self.motor_left.setPosition(np.inf)
            self.motor_right.setPosition(np.inf)
            for i in range(int(1000 * duration / self.timestep)):
                self.robot.step((self.timestep))
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            self.motor_left.setPosition(self.left_motor_position_sensor.getValue())
            self.motor_right.setPosition(self.right_motor_position_sensor.getValue())
        elif self.robot_interface == 'jetbot':
            self.robot.set_motors(self.robot_speed, self.robot_speed)
            for i in range(int(duration / self.timestep)):
                time.sleep(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            self.robot.stop()
        else:
            assert False

    def _translate_backward(self, duration=2, visualizer_fn=None):
        """
        Moves the robot backwards for duration timeslots
        @param duration: number of timeslots for which the motors will operate
        """

        if self.robot_interface == 'webots':
            self.motor_left.setPosition(-9999)
            self.motor_right.setPosition(-9999)
            for i in range(int(1000 * duration / self.timestep)):
                self.robot.step((self.timestep))
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())

            self.motor_left.setPosition(self.left_motor_position_sensor.getValue())
            self.motor_right.setPosition(self.right_motor_position_sensor.getValue())
        elif self.robot_interface == 'jetbot':
            self.robot.set_motors(-self.robot_speed, -self.robot_speed)
            for i in range(int(duration / self.timestep)):
                time.sleep(self.timestep)
                if visualizer_fn:
                    visualizer_fn(self.get_camera_data())
            self.robot.stop()
        else:
            assert False
