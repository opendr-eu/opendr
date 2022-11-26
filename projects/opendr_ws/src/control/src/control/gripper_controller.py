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
import actionlib
import franka_gripper.msg
from control_msgs.msg import GripperCommandActionGoal, GripperCommandGoal, GripperCommand
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

class Gripper():

    def __init__(self):
        self.move_client   = actionlib.SimpleActionClient('/franka_gripper/move', franka_gripper.msg.MoveAction)
        self.stop_client   = actionlib.SimpleActionClient('/franka_gripper/homing', franka_gripper.msg.StopAction)
        self.grasp_client  = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', franka_gripper.msg.HomingAction)

        self.controller_switcher = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        self.generic_grasp_client =  rospy.Publisher('/franka_gripper/gripper_action/goal', GripperCommandActionGoal, queue_size=10)

        self.move_client.wait_for_server()
        rospy.loginfo("Move gripper client connected")
        self.stop_client.wait_for_server()
        rospy.loginfo("Stop gripper client connected")
        self.grasp_client.wait_for_server()
        rospy.loginfo("Grasping client connected")
        self.homing_client.wait_for_server()
        rospy.loginfo("Homing gripper client connected")

    def switchToArmNavigationControl(self):
        rospy.loginfo('Switching to arm navigation control')
        switch_msg = SwitchControllerRequest()
        # switch_msg.start_controllers = ["cartesian_pose_controller"]
        # switch_msg.start_controllers = ["position_joint_trajectory_controller"]
        switch_msg.stop_controllers = ["cartesian_impedance_controller"]
        switch =  self.controller_switcher(switch_msg)
        print(switch.ok)
        return switch.ok

    def homing(self):
        goal = franka_gripper.msg.HomingActionGoal(goal={})
        self.homing_client.send_goal(goal)
        self.homing_client.wait_for_result()
        return self.homing_client.get_result()

    def move(self, speed=20.0, width=0.8):
        print(width)
        self.move_client.send_goal(
            franka_gripper.msg.MoveGoal(
                width,
                speed
            )
        )
        self.move_client.wait_for_result()
        return self.move_client.get_result()

    def stop(self):
        pass

    def grasp(self, force, width, speed=20.0, epsilon_inner=0.001, epsilon_outer=0.001):
        ''' width, epsilon_inner, epsilon_outer, speed, force '''
        print("""
            Grasping with
            {}
            {}
        """.format(force, width))
        self.grasp_client.send_goal(
            franka_gripper.msg.GraspGoal(
                width,
                franka_gripper.msg.GraspEpsilon(
                    epsilon_inner,
                    epsilon_outer
                ),
                speed,
                force
            )
        )
        self.grasp_client.wait_for_result()
        return self.grasp_client.get_result()

    def grasp2(self, width, force):
        ''' width, epsilon_inner, epsilon_outer, speed, force '''
        grasp_goal = GripperCommandGoal()
        grasp_goal.command.position = float(width)
        grasp_goal.command.max_effort = float(force)
        grasp_msg = GripperCommandActionGoal(goal=grasp_goal)
        self.generic_grasp_client.publish(grasp_msg)

    def grasp_triggered(self, msg):
        #self.grasp(50.0, 0.035)
        self.switchToArmNavigationControl()



