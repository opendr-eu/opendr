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
import argparse

import rospy 

from control.srv import *


def handle_move(request, gripper):
    speed = request.speed if request.speed > 0 else 20.0
    # width = request.width/100 if request.width > 0 else 0.05
    width = request.width
    gripper.move(speed, width)
    return MoveGripperResponse(True)

def handle_grasp(request, gripper):
    force = request.force if request.force > 0 else 70.0
    #width = request.width/100 if request.width > 0 else 0.004
    width = request.width # 0.008
    gripper.grasp(force, width)
    # robot.grasp2(width, force)
    return GraspResponse(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, help='MoveIt group name of the robot to control')
    args = parser.parse_args()

    rospy.init_node('gripper_control', anonymous=True)

    gripper = Gripper()
    # lambda msg: grasp(msg,panda_gripper, action_performed_pub)
    move_gripper_service = rospy.Service('/opendr/move_gripper', MoveGripper, handle_move, (gripper))
    grasping_service = rospy.Service('/opendr/grasp', Grasp, handle_grasp, (gripper) )

    rospy.loginfo("Gripper control node started!")

    rospy.spin()