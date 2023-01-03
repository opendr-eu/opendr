#!/usr/bin/env python

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

"""universal_robot_ros controller."""

import argparse
import rospy

from controller import Robot
from joint_state_publisher import JointStatePublisher
from gripper_command import GripperCommander
from trajectory_follower import TrajectoryFollower
from camera_publisher import CameraPublisher
from rosgraph_msgs.msg import Clock


class ROSController():

    def __init__(self, robot=None):
        if not robot:
            self.robot = Robot()
        else:
            self.robot = robot

        jointPrefix = rospy.get_param('prefix', '')
        if jointPrefix:
            print('Setting prefix to %s' % jointPrefix)

        self.jointStatePublisher = JointStatePublisher(self.robot, jointPrefix, '/')
        self.gripperCommander = GripperCommander(self.robot, self.jointStatePublisher, jointPrefix, 'panda')
        self.trajectoryFollower = TrajectoryFollower(self.robot, self.jointStatePublisher, jointPrefix, '/')
        self.cameraPublisher = CameraPublisher(self.robot, jointPrefix, '/')
        self.gripperCommander.start()
        self.trajectoryFollower.start()

        init_pos = {
            'panda_joint1': 0.000,
            'panda_joint2': -0.785,
            'panda_joint3': 0.0,
            'panda_joint4': -1.570,
            'panda_joint5': 0.0,
            'panda_joint6': 1.047,
            'panda_joint7': 0.0
        }

        for jt in init_pos:
            self.robot.getDevice(jt).setPosition(init_pos[jt])

        print("Robot sent to init pose")
        # we want to use simulation time for ROS
        self.clockPublisher = rospy.Publisher('clock', Clock, queue_size=1)
        if not rospy.get_param('use_sim_time', False):
            rospy.logwarn('use_sim_time is not set!')
        print("Clock publisher created")

    def run(self):
        timestep = int(self.robot.getBasicTimeStep())
        print("Entered thread")
        while self.robot.step(timestep) != -1 and not rospy.is_shutdown():
            self.jointStatePublisher.publish()
            self.cameraPublisher.publish()
            self.gripperCommander.update()
            self.trajectoryFollower.update()
            # pulish simulation clock
            msg = Clock()
            time = self.robot.getTime()
            msg.clock.secs = int(time)
            # round prevents precision issues that can cause problems with ROS timers
            msg.clock.nsecs = int(round(1000 * (time - msg.clock.secs)) * 1.0e+6)
            self.clockPublisher.publish(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-name', dest='nodeName', default='panda', help='Specifies the name of the node.')
    arguments, unknown = parser.parse_known_args()

    rospy.init_node(arguments.nodeName, disable_signals=True)

    controller = ROSController()
    controller.run()
    rospy.spin()
