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

from controller import Robot, Camera, Lidar

MAX_SPEED = 6.28

# create the instances.

robot = Robot()
cameraTop = Camera("front_top_camera")
cameraBottom = Camera("front_bottom_camera")
cameraRear = Camera("rear_camera")
cameraDownwards1 = Camera("camera_downwards1")
cameraDownwards2 = Camera("camera_downwards2")
cameraDownwards3 = Camera("camera_downwards3")
cameraDownwards4 = Camera("camera_downwards4")
lidar = Lidar("velodyne")
gps = robot.getDevice("Hemisphere_v500")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

leftMotorFront = robot.getDevice("left_front_wheel_joint_motor")
leftMotorRear = robot.getDevice("left_rear_wheel_joint_motor")
rightMotorFront = robot.getDevice("right_front_wheel_joint_motor")
rightMotorRear = robot.getDevice("right_rear_wheel_joint_motor")

# enable all the devices
cameraTop.enable(4 * timestep)
cameraBottom.enable(10 * timestep)
# cameraBottom.recognitionEnable(10 * timestep)
cameraRear.enable(10 * timestep)
robot.step(2 * timestep)
cameraDownwards1.enable(10 * timestep)
robot.step(2 * timestep)
cameraDownwards2.enable(10 * timestep)
robot.step(2 * timestep)
cameraDownwards3.enable(10 * timestep)
robot.step(2 * timestep)
cameraDownwards4.enable(10 * timestep)
lidar.enable(timestep)
lidar.enablePointCloud()
gps.enable(timestep)

# set motor veloctiy control
leftMotorFront.setPosition(float('inf'))
leftMotorRear.setPosition(float('inf'))
rightMotorFront.setPosition(float('inf'))
rightMotorRear.setPosition(float('inf'))
leftMotorFront.setVelocity(0.1 * MAX_SPEED)
leftMotorRear.setVelocity(0.1 * MAX_SPEED)
rightMotorFront.setVelocity(0.1 * MAX_SPEED)
rightMotorRear.setVelocity(0.1 * MAX_SPEED)

while robot.step(timestep) != -1:
    pass
