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

from controller import Robot
import pathlib
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


DATASET_NAME = 'dataset_location/UAV'
STOP_ON = 193

useMavic = True


# useMavic=False

class Mavic(Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # 68.5  # with this thrust, the drone lifts.

    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6  # 0.3 #0.6
    K_VERTICAL_P = 3.0  # 1.0 #3.0        # P constant of the vertical PID.
    K_ROLL_P = 12.5  # 25.0 #50.0           # P constant of the roll PID.
    K_PITCH_P = 12.5  # 15.0 #30.0          # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 1.0
    MAX_PITCH_DISTURBANCE = -1

    # Precision between the target position and the robot position in meters
    target_precision = 1.0

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.camera.recognitionEnable(self.time_step)
        self.camera.enableRecognitionSegmentation()

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """

        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=True, verbose_target=False):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in
                zip(self.target_position, self.current_pose[0:2])]):
            # test
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])
                self.target_altitude = 0
                self.target_altitude = 5

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])

        angle_left = self.target_position[2] - self.current_pose[5]

        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non-proportional and decreasing function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                    (self.target_position[1] - self.current_pose[1]) ** 2))
            print("target_position 0: {:.4f}, current_pose 0: {:.4f}".format(
                self.target_position[0], self.current_pose[0]))
            print("target_position 1: {:.4f}, current_pose 1: {:.4f}".format(
                self.target_position[1], self.current_pose[1]))
            print("target angle: {:.4f}, current angle: {:.4f}".format(
                self.target_position[2], self.current_pose[5]))
            print("target angle real: {:.4f}, current angle: {:.4f}".format(
                self.target_position[3], self.current_pose[5]))

            print("target_position: {}, current_pose: {}".format(
                self.target_position, self.current_pose))

            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def save_device_measurements(self, index, second, objects, static_objects):

        index = index.zfill(6)

        number = int(index)

        index = str(second) + '_' + str(index)

        if (number % 3 == 0):

            # RGB Camera images
            dir_name = os.path.join(DATASET_NAME, self.camera.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            self.camera.saveImage(os.path.join(dir_name, index + ".jpg"), 100)

            dir_name = os.path.join(DATASET_NAME, 'annotations', self.camera.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            self.camera.saveRecognitionSegmentationImage(os.path.join(dir_name, index + "_segmented.jpg"), 100)

            with open(os.path.join(dir_name, index + "_annotations.txt"), 'w') as f:
                f.write('# YOLO annotations format: <object-class> <x> <y> <width> <height>\n')
                for object in self.camera.getRecognitionObjects():
                    position = object.getPositionOnImage()
                    size = object.getSizeOnImage()
                    f.write('"{}" {} {} {} {}\n'.format(object.getModel(), position[0], position[1], size[0], size[1]))

                    # print(object.getModel()),

        if (number % 20 == 0):
            # GPS
            dir_name = os.path.join(DATASET_NAME, self.gps.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
                value = self.gps.getValues()
                f.write(f'{value[0]} {value[1]} {value[2]}')

        # IMU
        dir_name = os.path.join(DATASET_NAME, self.imu.getName())
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
            value = self.imu.getRollPitchYaw()
            f.write(f'{value[0]} {value[1]} {value[2]}')

        if (number % 3 == 0):

            # Objects position
            dir_name = os.path.join(DATASET_NAME, 'annotations', 'scene')
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
                for object in objects:
                    f.write(f'"{object}" \n')

                for object in static_objects:
                    position = object['position']
                    type_name = object['type name']
                    f.write(f'"{type_name}" {position[0]} {position[1]} {position[2]}\n')

    def run(self):
        t1 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # Specify the patrol coordinates
        waypoints = [[-45, -8, 5, 0.065], [-35, -8, 5, 0.065], [-25, -8, 5, 0.065], [-15, -8, 5, 0.065],
                     [-5, -8, 5, 0.065], [+5, -8, 5, 1.57 + 0.065],
                     [+5, -2, 5, 1.57 + 0.065], [+5, -2, 5, 1.57 + 0.065], [+5, -2, 5, 1.57 + 0.065],
                     [+5, -2, 5, 3.14 + 0.065],
                     [-10, -0, 5, 3.14 + 0.065], [-20, -0, 5, 3.14 + 0.065], [-30, -0, 5, 3.14 + 0.065],
                     [-40, -0, 5, 3.14 + 0.065], [-50, -0, 5, 3.14 + 0.065],
                     [-52, 6.0, 5, 1.57 + 0.065], [-35, 8.0, 5, 1.57 + 0.065], [-25, 8.0, 5, 1.57 + 0.065],
                     [-15, 8.0, 5, 1.57 + 0.065], [+5, 8.0, 5, 1.57 + 0.065]]

        # target altitude of the robot in meters
        self.target_altitude = 15
        self.target_altitude = 5

        data_index = 0
        index = 1
        second = 1

        obstacle_classes = ['CharacterSkin', 'Cat', 'Cow', 'Deer', 'Dog', 'Fox', 'Horse', 'Rabbit', 'Sheep']

        # retrieve static objects
        static_objects = []

        object_classes = ['AgriculturalWarehouse', 'Barn', 'Tractor', 'BungalowStyleHouse', 'HouseWithGarage', 'Silo',
                          'SimpleTree',
                          'Forest', 'PicketFenceWithDoor', 'PicketFence', 'Forest', 'Road', 'StraightRoadSegment',
                          'RoadIntersection']
        object_classes += obstacle_classes

        MAX_RECORDS_PER_SCENARIO=19300

        while (self.step(self.time_step) != -1) and index < MAX_RECORDS_PER_SCENARIO:

            # Read sensors
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()

            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            if altitude > self.target_altitude - 1:

                # as soon as it reach the target altitude, compute the disturbances to go to the given waypoints.
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(
                        waypoints)
                    t1 = self.getTime()
                    # print('moving to target')
                    # print('altitude: {}, target altitude: {}'.format(altitude, self.target_altitude))
                    # print('x_pos: {}, y_pos: {}, altitude: {}'.format(x_pos, y_pos, altitude))

            # else:
            # print('altitude: {}, target altitude: {}'.format(altitude, self.target_altitude))

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

            objects = []
            for object in self.camera.getRecognitionObjects():
                objects.append(object.getModel())

            self.save_device_measurements(str(data_index), str(second), objects,
                                          static_objects)

            # print('mavic 357')
            data_index += 1
            index += 1
            if (index % 100 == 0):
                print(f'Mavic second: {second}')
                second += 1
                index = 1
                data_index = 1

                if (second == STOP_ON):
                    index = MAX_RECORDS_PER_SCENARIO


# To use this controller, the basicTimeStep should be set to 8 and the defaultDamping
# with a linear and angular damping both of 0.5

if (useMavic):
    robot = Mavic()
    robot.run()
