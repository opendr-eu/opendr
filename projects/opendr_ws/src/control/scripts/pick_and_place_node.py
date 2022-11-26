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


import rospy
from control.srv import *
from control.pick_and_place_server import PickAndPlaceServer


def build_srv_name(namespace, body):
    return '/' + namespace + '/' + body

def start_pick_and_place():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, help='Namespace of the robot to control')
    parser.add_argument('--gripper', type=str, help='Namespace of the gripper to control')
    args = parser.parse_args()

    rospy.init_node('opendr_pick_and_place_server', anonymous=False)  # initialize ros node

    arm_srvs = ['set_joint_state', 'set_pose_target', 'set_pose_target_1D', 'set_pose_target_2D', 'rotate_ee', 'stop_action', 'resume_action'] 
    gripper_srvs = ['move_gripper', 'grasp'] 

    arm_srvs = [build_srv_name(args.arm, x) for x in arm_srvs]
    gripper_srvs = [build_srv_name(args.gripper, x) for x in gripper_srvs]

    for i in arm_srvs:
        rospy.wait_for_service(i)

    for i in gripper_srvs:   
        rospy.wait_for_service(i)

    rotate_EE = rospy.ServiceProxy('rotate_ee', RotateEE)
    stop_action = rospy.ServiceProxy('stop_action', Trigger)
    resume_action = rospy.ServiceProxy('resume_action', Trigger)
    move_joint_space = rospy.ServiceProxy('set_joint_state', SetJointState)
    move_cartesian_space = rospy.ServiceProxy('set_pose_target', SetPoseTarget)
    move_cartesian_space_1D = rospy.ServiceProxy('set_pose_target_1D', SetPoseTarget1D)
    move_cartesian_space_2D = rospy.ServiceProxy('set_pose_target_2D', SetPoseTarget2D)

    grasp = rospy.ServiceProxy('grasp', Grasp)
    move_gripper = rospy.ServiceProxy('move_gripper', MoveGripper)
    
    pick_and_place_server = PickAndPlaceServer(args.arm, args.gripper)
    pick_and_place_server.start()

    def stop_pick_and_place_server():
        pick_and_place_server.stop()

    rospy.on_shutdown(stop_pick_and_place_server)

    rospy.spin() 


if __name__ == '__main__':
    start_pick_and_place()