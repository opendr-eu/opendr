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

import rospy
from control.srv import MoveGripper, MoveGripperResponse, Grasp, GraspResponse
from control.gripper_controller import Gripper


def handle_move(request, gripper):
    success = True
    try:
        speed = 50.0
        # width = request.width/100 if request.width > 0 else 0.05
        width = request.width
        gripper.move(width, speed)
    except Exception:
        success = False
    return MoveGripperResponse(success=success)


def handle_grasp(request, gripper):
    success = True
    try:
        force = request.force if request.force > 0 else 70.0
        # width = request.width/100 if request.width > 0 else 0.004
        width = request.width  # 0.008
        gripper.grasp(force, width)
        # robot.grasp2(width, force)
    except Exception:
        success = False
    return GraspResponse(success=success)


if __name__ == '__main__':
    rospy.init_node('gripper_control', anonymous=True)

    gripper = Gripper()
    move_gripper_service = rospy.Service('/opendr/move_gripper', MoveGripper, lambda msg: handle_move(msg, gripper))
    grasping_service = rospy.Service('/opendr/grasp', Grasp, lambda msg: handle_grasp(msg, gripper))

    rospy.loginfo("Gripper control node started!")

    rospy.spin()
