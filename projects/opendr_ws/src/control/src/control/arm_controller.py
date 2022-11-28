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

import sys
import copy
import math
import argparse
import moveit_commander

import tf
import rospy
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerResponse
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control.srv import *


class RobotController:

    def __init__(self, moveit_group):
        self.group = moveit_group
        self.gripper = Gripper()
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_max_acceleration_scaling_factor(0.1)

        self._last_goal = self.group.get_current_pose().pose

    def rotate_ee(self, angle):
        '''
        45 is neutral math.pi/4
        -45 rotate to the left -math.pi/4
        135 rotate to the right 3*math.pi/4
        '''
        joint_goal = self.group.get_current_joint_values()
        yaw = angle + math.pi/4
        if yaw > 3*math.pi/4:
            yaw = yaw - math.pi
        elif yaw < -math.pi/4:
            yaw = yaw + math.pi
        joint_goal[6] = yaw
        self.group.go(joint_goal, wait=True)
        self.stop()

    def modify_plan(self, plan, speed_factor=0.1):
        new_plan = RobotTrajectory()
        new_plan.joint_trajectory.joint_names = plan.joint_trajectory.joint_names
        new_plan.joint_trajectory.header = plan.joint_trajectory.header
        for p in plan.joint_trajectory.points:
            new_p = JointTrajectoryPoint()
            new_p.time_from_start = p.time_from_start/speed_factor
            new_p.positions = p.positions
            for i in range(len(p.velocities)):
                new_p.velocities.append(p.velocities[i]*speed_factor)
            for i in range(len(p.accelerations)):
                new_p.accelerations.append(p.accelerations[i]*speed_factor)
            new_plan.joint_trajectory.points.append(new_p)
        return new_plan

    def move_to_joint_target(self, joint_values):
        self.group.go(list(joint_values), wait=True)
        self.stop()

    def move_to_cartesian_target(self, cartesian_values):
        """
        cartesian_values: 7 dimensional vecotr, [cartesian_position, cartesian_Quaternion]
                          [x, y, z, quat_x, quat_y, quat_z, quat_w]
        """
        if isinstance(cartesian_values, list):
            pose_goal = Pose()
            pose_goal.orientation.x = cartesian_values[-4]
            pose_goal.orientation.y = cartesian_values[-3]
            pose_goal.orientation.z = cartesian_values[-2]
            pose_goal.orientation.w = cartesian_values[-1]
            pose_goal.position.x = cartesian_values[0]
            pose_goal.position.y = cartesian_values[1]
            pose_goal.position.z = cartesian_values[2]
        else:
            pose_goal = cartesian_values
        self._last_goal = pose_goal
        self.group.set_pose_target(pose_goal)
        self.group.go(wait=True)
        self.stop()


    def move_to_2D_cartesian_target(self, pose, slow=False):
        waypoints = []
        next_point = self.group.get_current_pose().pose
        next_point.position.x = pose[0]
        next_point.position.y = pose[1]
        self._last_goal = next_point
        waypoints.append(copy.deepcopy(next_point))
        (plan, fraction) = self.group.compute_cartesian_path(
                               waypoints,   # waypoints to follow
                               0.01,        # eef_step
                               0.0)         # jump_threshold
        if slow:
            plan = self.modify_plan(plan)
        self.group.execute(plan, wait=True)
        self.stop()

    def plan_linear_z(self, dist, slow=False):
        waypoints = []
        next_point = self.group.get_current_pose().pose
        next_point.position.z = dist
        self._last_goal = next_point
        waypoints.append(copy.deepcopy(next_point))
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0)
        if slow:
            plan = self.modify_plan(plan)
        self.group.execute(plan)
        self.stop()

    def stop(self):
        self.group.stop()
        self.group.clear_pose_targets()

    def resume(self):
        print(type(Pose()))
        print(type(self._last_goal))
        if type(self._last_goal) == type(Pose()):
            self.move_to_cartesian_target(self._last_goal)
        else:
            self.move_to_joint_target(self._last_goal)
        self._last_goal = self.group.get_current_pose().pose
