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


import sys
import copy
import rospy
import numpy as np
import time
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
from math import pi
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import Int16, Float32MultiArray
from geometry_msgs.msg import PoseStamped


def all_close(goal, actual, tolerance):

    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class SingleDemoGraspAction(object):

    """MoveIt_Commander"""
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)
        hand_grp = moveit_commander.MoveGroupCommander("hand")
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        rospy.Subscriber("/commands", Float32MultiArray, self.callback_cmd)
        request_publisher = rospy.Publisher('/request_detection', Int16, queue_size=10)

        planning_frame = group.get_planning_frame()
        eef_link = group.get_end_effector_link()
        group_names = robot.get_group_names()

        self.robot = robot
        self.scene = scene
        self.group = group
        self.hand_grp = hand_grp
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.request_publisher = request_publisher
        self.last_msg_idx = 1e10
        self.last_msg_used = 1e10
        self.detections = [1e10, 1e10, 1e10, 1e10, 1e10]
        self.camera_focal = 550

    # callback to receive commands in x-y plane
    def callback_cmd(self, data):
        # global xy_cmd
        if data.data[3] != 1e10:

            rotcmd = data.data[3]
            if (rotcmd > 180):
                rotcmd = -360 + rotcmd
            if (rotcmd > 0 and rotcmd > 90):
                rotcmd = -180 + rotcmd
            if (rotcmd < 0 and rotcmd < -90):
                rotcmd = 180 + rotcmd

            self.last_msg_idx = data.data[0]
            self.detections = np.asarray([data.data[1], data.data[2], rotcmd,
                                         data.data[4], data.data[5]])
        else:
            self.last_msg_idx = data.data[0]
            self.detections = np.asarray([1e10, 1e10, 1e10, 1e10, 1e10])

    def home_pose(self):

        self.go_to_joint_state((-0.00028, -0.391437, -0.000425, -1.178905,
                                0.000278, 0.785362, 0.8))
        self.gripper_move((-8.584933242598848e-05, 0.00019279284))
        self.open_hand()
        cartesian_plan, fraction = self.plan_linear_x(0.17)
        self.execute_plan(cartesian_plan)
        cartesian_plan, fraction = self.plan_linear_z(-0.3)
        self.execute_plan(cartesian_plan)

    def open_hand(self, wait=True):
        joint_goal = self.hand_grp.get_current_joint_values()
        joint_goal[0] = 0.0399

        self.hand_grp.set_goal_joint_tolerance(0.001)
        self.hand_grp.go(joint_goal, wait=wait)

        self.hand_grp.stop()

        current_joints = self.hand_grp.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def close_hand(self, wait=True):
        joint_goal = self.hand_grp.get_current_joint_values()
        joint_goal[0] = 0.0001

        self.hand_grp.set_goal_joint_tolerance(0.001)
        self.hand_grp.go(joint_goal, wait=wait)

        if (not wait):
            return

        self.hand_grp.stop()
        current_joints = self.hand_grp.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    # function to control the gripper by setting width
    def gripper_move(self, cmd):

        group = self.hand_grp
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = cmd[0]
        joint_goal[1] = cmd[1]
        group.go(joint_goal, wait=True)
        group.stop()
        current_joints = self.hand_grp.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    # joint space
    def go_to_joint_state(self, cmd):

        group = self.group
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = cmd[0]
        joint_goal[1] = cmd[1]
        joint_goal[2] = cmd[2]
        joint_goal[3] = cmd[3]
        joint_goal[4] = cmd[4]
        joint_goal[5] = cmd[5]
        joint_goal[6] = cmd[6]
        group.go(joint_goal, wait=True)
        group.stop()
        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    # cartesian target
    def go_to_pose_goal(self, target):

        group = self.group
        group.get_current_pose().pose
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = target.pose.orientation.x
        pose_goal.orientation.y = target.pose.orientation.y
        pose_goal.orientation.z = target.pose.orientation.z
        pose_goal.orientation.w = target.pose.orientation.w
        pose_goal.position.x = target.pose.position.x
        pose_goal.position.y = target.pose.position.y
        pose_goal.position.z = target.pose.position.z
        print(pose_goal)
        group.set_pose_target(pose_goal)
        group.go(wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    # get cartesian pose
    def get_current_pose(self):

        group = self.group
        wpose = group.get_current_pose().pose
        return wpose

    # linear movemonet planning along x axis of the reference frame
    def plan_linear_x(self, dist):

        group = self.group
        waypoints = []
        wpose = group.get_current_pose().pose
        wpose.position.x += dist
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan, fraction

    # linear movemonet planning along y axis of the reference frame
    def plan_linear_y(self, dist):

        group = self.group
        waypoints = []
        wpose = group.get_current_pose().pose
        wpose.position.y += dist
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan, fraction

    # linear movemonet planning along z axis of the reference frame
    def plan_linear_z(self, dist):

        group = self.group
        waypoints = []
        wpose = group.get_current_pose().pose
        wpose.position.z += dist
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan, fraction

    # execute planned movements
    def execute_plan(self, plan):

        group = self.group
        group.execute(plan, wait=True)

    # rotation in camera frame (not only last revolut joint)
    def fix_angle(self, rotcmd):

        ps = listener_tf.lookupTransform('/camera_optical_frame', '/panda_link8', rospy.Time(0))
        my_point = PoseStamped()
        my_point.header.frame_id = "camera_optical_frame"
        my_point.header.stamp = rospy.Time(0)
        my_point.pose.position.x = ps[0][0]
        my_point.pose.position.y = ps[0][1]
        my_point.pose.position.z = ps[0][2]

        theta = rotcmd / 180 * pi
        quat_rotcmd = tf.transformations.quaternion_from_euler(theta, 0, 0)
        quat = tf.transformations.quaternion_multiply(quat_rotcmd, ps[1])

        my_point.pose.orientation.x = quat[0]
        my_point.pose.orientation.y = quat[1]
        my_point.pose.orientation.z = quat[2]
        my_point.pose.orientation.w = quat[3]

        ps = listener_tf.transformPose("/panda_link0", my_point)
        self.go_to_pose_goal(ps)

    # using center of keypoints as target pose, translates the 2D coordinate of
    # Target pose in camera frame into world frame and execute grasping action in 3D
    def reach_grasp_hover_kps(self):

        (trans1, rot1) = listener_tf.lookupTransform('/panda_link0', '/camera_optical_frame', rospy.Time(0))
        z_to_surface = trans1[2]
        to_world_scale = z_to_surface / self.camera_focal
        x_dist = self.detections[3] * to_world_scale
        y_dist = self.detections[4] * to_world_scale
        my_point = PoseStamped()
        my_point.header.frame_id = "camera_optical_frame"
        my_point.header.stamp = rospy.Time(0)
        my_point.pose.position.x = 0
        my_point.pose.position.y = -x_dist
        my_point.pose.position.z = y_dist  # 0.1
        theta = 0
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        my_point.pose.orientation.x = quat[0]
        my_point.pose.orientation.y = quat[1]
        my_point.pose.orientation.z = quat[2]
        my_point.pose.orientation.w = quat[3]
        ps = listener_tf.transformPose("/panda_link0", my_point)

        my_point1 = PoseStamped()
        my_point1.header.frame_id = "/panda_link8"
        my_point1.header.stamp = rospy.Time(0)
        my_point1.pose.position.x = 0
        my_point1.pose.position.y = 0
        my_point1.pose.position.z = 0
        theta1 = 0
        quat1 = tf.transformations.quaternion_from_euler(0, 0, theta1)
        my_point1.pose.orientation.x = quat1[0]
        my_point1.pose.orientation.y = quat1[1]
        my_point1.pose.orientation.z = quat1[2]
        my_point1.pose.orientation.w = quat1[3]
        ps1 = listener_tf.transformPose("/panda_link0", my_point1)
        ps1.pose.position.x = ps.pose.position.x
        ps1.pose.position.y = ps.pose.position.y
        ps1.pose.position.z = ps.pose.position.z

        self.go_to_pose_goal(ps1)
        print("reached pose")
        print(self.get_current_pose())
        self.approach_grasp()
        self.close_hand()

    # Using the center of bounding box as target, approach the target pose and translates
    # 2D to 3D from image frame to world frame respectively
    def reach_hover(self):

        print("self.detections")
        print(self.detections)

        (trans1, rot1) = listener_tf.lookupTransform('/panda_link0', '/camera_optical_frame', rospy.Time(0))
        z_to_surface = trans1[2]
        to_world_scale = z_to_surface / self.camera_focal

        x_dist = self.detections[0] * to_world_scale
        y_dist = self.detections[1] * to_world_scale

        my_point = PoseStamped()
        my_point.header.frame_id = "camera_optical_frame"
        my_point.header.stamp = rospy.Time(0)
        my_point.pose.position.x = 0
        my_point.pose.position.y = -x_dist
        my_point.pose.position.z = y_dist
        theta = 0
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        my_point.pose.orientation.x = quat[0]
        my_point.pose.orientation.y = quat[1]
        my_point.pose.orientation.z = quat[2]
        my_point.pose.orientation.w = quat[3]
        ps = listener_tf.transformPose("/panda_link0", my_point)

        (trans, rot) = listener_tf.lookupTransform('/panda_link0', '/camera_optical_frame', rospy.Time(0))
        data = (ps.pose.position.x - trans[0], ps.pose.position.y - trans[1])

        cartesian_plan, fraction = self.plan_linear_x(data[0])
        self.execute_plan(cartesian_plan)
        cartesian_plan, fraction = self.plan_linear_y(data[1])
        self.execute_plan(cartesian_plan)

    def approach_grasp(self):

        my_point = PoseStamped()
        my_point.header.frame_id = "camera_optical_frame"
        my_point.header.stamp = rospy.Time(0)
        my_point.pose.position.x = 0.48
        my_point.pose.position.y = 0.005  # TODO y element should be "0"
        my_point.pose.position.z = 0
        theta = 0
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        my_point.pose.orientation.x = quat[0]
        my_point.pose.orientation.y = quat[1]
        my_point.pose.orientation.z = quat[2]
        my_point.pose.orientation.w = quat[3]
        ps = listener_tf.transformPose("/panda_link0", my_point)

        (trans, rot) = listener_tf.lookupTransform('/panda_link0', '/camera_optical_frame', rospy.Time(0))
        data = (ps.pose.position.x - trans[0], ps.pose.position.y - trans[1], ps.pose.position.z - trans[2])
        cartesian_plan, fraction = self.plan_linear_x(data[0])
        self.execute_plan(cartesian_plan)
        cartesian_plan, fraction = self.plan_linear_y(data[1])
        self.execute_plan(cartesian_plan)
        cartesian_plan, fraction = self.plan_linear_z(data[2])
        self.execute_plan(cartesian_plan)

    # sending a rquesst message to detection server and waits for a detection reply
    # TODO: change the request request method to a more robust one
    def request_detection(self):
        while (True):
            msg_identifier = random.randint(0, 1000)
            print(msg_identifier)
            if msg_identifier == self.last_msg_used:
                print("new id not generated")
                continue
            else:
                print("new id generated")
                break
        print(msg_identifier)
        self.request_publisher.publish(msg_identifier)

        print("waiting for detection to be received:")
        start_time = time.time()

        while (time.time() - start_time < 10):

            print("wait for the message")
            time.sleep(1)
            if self.last_msg_idx != self.last_msg_used:
                self.last_msg_used = self.last_msg_idx
                print("detections received:")
                print(self.detections)
                break
            else:
                print("waiting for a reply...")


# executing a sequence of actions to demonstrate a grasping action based on
# single demo grasp model
def main():
    try:
        print("============ Initializing panda control ...")
        Commander = SingleDemoGraspAction()

        # reaching home position above objects with grippers opened
        print("============ Press `Enter` to reach home pose ...")
        input()
        Commander.home_pose()

        # send a request detection to detection server and waits until it receives a reply
        print("press enter to send a detection request to detector node")
        input()
        Commander.request_detection()

        # after receiving the detections, checks if there is an object found in the
        # image frame, and brings the robot's camera view above the object and correct
        # the object's orientation (in case some objects might be placed in the edges
        # and could be hardly visible) to have better predictions.
        if Commander.detections[2] != 1e10:
            Commander.reach_hover()
            Commander.fix_angle(Commander.detections[2])

        # another request, to receive the exact location of grasp followed by translating
        # the 2D coordinate (position in image frame) to 3D (corresponding position
        # in world frame) and executing the grasping action.
        print("============ Press `Enter` to find and reach grasp location")
        input()
        Commander.request_detection()
        if Commander.detections[0] != 1e10 or Commander.detections[1] != 1e10:
            Commander.reach_grasp_hover_kps()

        # lifting the object
        print("give input to close the gripper and lift the object")
        input()
        Commander.close_hand()
        cartesian_plan, fraction = Commander.plan_linear_z(0.45)
        Commander.execute_plan(cartesian_plan)
        print("============ Python Commander demo complete!")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':

    rospy.init_node('SingleDemoGraspAction', anonymous=True)
    listener_tf = tf.TransformListener()
    main()
    sys.exit()
