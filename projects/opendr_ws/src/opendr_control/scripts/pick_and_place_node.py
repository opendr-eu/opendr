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
from std_srvs.srv import Trigger
from control.srv import RotateEE, SetJointState, SetPoseTarget, SetPoseTarget2D, SetPoseTarget1D
from control.pick_and_place_server import PickAndPlaceServer


def build_srv_name(namespace, body):
    return '/' + namespace + '/' + body


def start_pick_and_place():
    rospy.init_node('opendr_pick_and_place_server', anonymous=False)  # initialize ros node

    arm_srvs = ['rotate_ee', 'stop_action', 'resume_action', 'set_joint_state',
                'set_pose_target', 'set_pose_target_1D', 'set_pose_target_2D']
    gripper_srvs = ['grasp', 'move_gripper']

    arm_srvs = [build_srv_name(rospy.get_param('/opendr_pick_and_place_server/arm'), x) for x in arm_srvs]
    gripper_srvs = [build_srv_name(rospy.get_param('/opendr_pick_and_place_server/gripper'), x) for x in gripper_srvs]

    for i in arm_srvs:
        rospy.wait_for_service(i)

    for i in gripper_srvs:
        rospy.wait_for_service(i)

    rotate_EE = rospy.ServiceProxy(arm_srvs[0], RotateEE)
    stop_action = rospy.ServiceProxy(arm_srvs[1], Trigger)
    resume_action = rospy.ServiceProxy(arm_srvs[2], Trigger)
    move_joint_space = rospy.ServiceProxy(arm_srvs[3], SetJointState)
    move_cartesian_space = rospy.ServiceProxy(arm_srvs[4], SetPoseTarget)
    move_cartesian_space_1D = rospy.ServiceProxy(arm_srvs[5], SetPoseTarget1D)
    move_cartesian_space_2D = rospy.ServiceProxy(arm_srvs[6], SetPoseTarget2D)

    grasp = rospy.ServiceProxy(gripper_srvs[0], Grasp)
    move_gripper = rospy.ServiceProxy(gripper_srvs[1], MoveGripper)

    pick_and_place_server = PickAndPlaceServer(rotate_EE, stop_action, resume_action,
                                               move_joint_space, move_cartesian_space,
                                               move_cartesian_space_1D, move_cartesian_space_2D,
                                               grasp, move_gripper)
    pick_and_place_server.start()

    def stop_pick_and_place_server():
        pick_and_place_server.stop()

    rospy.on_shutdown(stop_pick_and_place_server)

    rospy.spin()


if __name__ == '__main__':
    start_pick_and_place()
