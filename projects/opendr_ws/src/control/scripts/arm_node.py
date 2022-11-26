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


def handle_move_to_target(req):
    success = True
    try:
        self.move_to_joint_target(req.q)
    except:
        success = False
    return SetJointStateResponse(success=success)

def handle_move_to_cartesian_target(req):
    success = True
    try:        
        self.move_to_cartesian_target(req.pose)
    except:
        success = False
    return SetPoseTargetResponse(success=success)

def handle_stop(self, req):
    success = True
    try:
        self.stop()
    except:
        success = False
    return TriggerResponse(success=success)

def handle_resume(req):
    success = True
    try:
        self.resume()
    except Exception as e:
        print(e)
        success = False
    return TriggerResponse(success=success)

def handle_move_to_2D_cartesian_target(req):
    success = True
    try:
        self.move_to_2D_cartesian_target(req.point, req.slow)
    except:
        success = False
    return SetPoseTarget2DResponse(success=success)

def handle_move_to_1D_cartesian_target(req):
    success = True
    try:
        self.plan_linear_z(req.z, req.slow)
    except:
        success = False
    return SetPoseTarget1DResponse(success=success)

def handle_rotate_ee(req):
    success = True
    try:
        self.rotate_ee(req.angle)
    except:
        success = False
    return RotateEEResponse(success=success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, help='MoveIt group name of the robot to control')
    args = parser.parse_args()

    rospy.init_node('arm_control', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(args.group_name)

    robot_arm = RobotController(group)

    rotate_ee_service = rospy.Service('/opendr/rotate_ee', RotateEE, handle_rotate_ee)
    stop_action_service = rospy.Service('/opendr/stop_action', Trigger, handle_stop)
    resume_action_service = rospy.Service('/opendr/resume_action', Trigger, handle_resume)
    action_service = rospy.Service('/opendr/set_joint_state', SetJointState, handle_move_to_target)
    cartesian_action_service = rospy.Service('/opendr/set_pose_target', SetPoseTarget, handle_move_to_cartesian_target)
    cartesian_action_service_2d = rospy.Service('/opendr/set_pose_target_2D', SetPoseTarget2D, handle_move_to_2D_cartesian_target)
    cartesian_action_service_1d = rospy.Service('/opendr/set_pose_target_1D', SetPoseTarget1D, handle_move_to_1D_cartesian_target)

    rospy.loginfo("Arm control node started!")

    rospy.spin()