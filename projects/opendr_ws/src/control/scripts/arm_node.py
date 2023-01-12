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


import sys
import moveit_commander

import rospy 
from std_srvs.srv import Trigger, TriggerResponse

from control.arm_controller import RobotController
from control.srv import (SetJointStateResponse, SetPoseTargetResponse, SetPoseTarget2DResponse,
                         SetPoseTarget1DResponse, RotateEEResponse, RotateEE, SetJointState, SetPoseTarget, 
                         SetPoseTarget2D, SetPoseTarget1D)


def handle_move_to_target(req, robot_arm):
    success = True
    try:
        robot_arm.move_to_joint_target(req.q)
    except Exception:
        success = False
    return SetJointStateResponse(success=success)


def handle_move_to_cartesian_target(req, robot_arm):
    success = True
    try:        
        robot_arm.move_to_cartesian_target(req.pose)
    except Exception:
        success = False
    return SetPoseTargetResponse(success=success)


def handle_stop(req, robot_arm):
    success = True
    try:
        robot_arm.stop(pause=True)
    except Exception:
        success = False
    return TriggerResponse(success=success)


def handle_resume(req, robot_arm):
    success = True
    try:
        robot_arm.resume()
    except Exception as e:
        print(e)
        success = False
    return TriggerResponse(success=success)


def handle_move_to_2D_cartesian_target(req, robot_arm):
    success = True
    try:
        robot_arm.move_to_2D_cartesian_target(req.point, req.slow)
    except Exception:
        success = False
    return SetPoseTarget2DResponse(success=success)


def handle_move_to_1D_cartesian_target(req, robot_arm):
    success = True
    try:
        robot_arm.plan_linear_z(req.z, req.slow)
    except Exception:
        success = False
    return SetPoseTarget1DResponse(success=success)


def handle_rotate_ee(req, robot_arm):
    success = True
    try:
        robot_arm.rotate_ee(req.angle)
    except Exception:
        success = False
    return RotateEEResponse(success=success)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, help='MoveIt group name of the robot to control')
    args = parser.parse_args()
    '''

    rospy.init_node('arm_control', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(rospy.get_param('/arm_control/group_name'))

    robot_arm = RobotController(group)

    rotate_ee_service = rospy.Service('/opendr/rotate_ee', RotateEE, lambda msg: handle_rotate_ee(msg, robot_arm))
    stop_action_service = rospy.Service('/opendr/stop_action', Trigger, lambda msg: handle_stop(msg, robot_arm))
    resume_action_service = rospy.Service('/opendr/resume_action', Trigger, lambda msg: handle_resume(msg, robot_arm))
    action_service = rospy.Service('/opendr/set_joint_state', SetJointState, lambda msg: handle_move_to_target(msg, robot_arm))
    cartesian_action_service = rospy.Service('/opendr/set_pose_target', SetPoseTarget, 
                                             lambda msg: handle_move_to_cartesian_target(msg, robot_arm))
    cartesian_action_service_2d = rospy.Service('/opendr/set_pose_target_2D', SetPoseTarget2D, 
                                                lambda msg: handle_move_to_2D_cartesian_target(msg, robot_arm))
    cartesian_action_service_1d = rospy.Service('/opendr/set_pose_target_1D', SetPoseTarget1D, 
                                                lambda msg: handle_move_to_1D_cartesian_target(msg, robot_arm))

    rospy.loginfo("Arm control node started!")

    rospy.spin()
    