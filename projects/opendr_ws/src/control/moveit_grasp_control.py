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
import argparse
import moveit_commander


class RobotControlNode(object):

    """MoveIt_Commander"""
    def __init__(self, group):
        self.group = group
        self.group.set_max_velocity_scaling_factor(0.2)

    def rotate_ee(self, angle):
        try:     
            wpose = self.group.get_current_pose().pose
            roll, pitch, yaw = tf.transformations.euler_from_quaternion([wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w])
            print("Original yaw - {}".format(yaw))
            #if angle < 0: 
            #    angle = angle + math.pi/2
            yaw = angle + math.pi/4 
            print("Intermediate yaw - {}".format(yaw))
            print("comparison {}".format(-math.pi + math.pi/4))
            if yaw > math.pi/4:
                yaw = yaw - math.pi
            print("New yaw - {}".format(yaw))
            quat_rotcmd = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            #q_tf = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
            #q_rot = quaternion_from_euler(0, -math.pi, 2*math.pi)
            #quat = quaternion_multiply(q_rot, q_tf)
            wpose.orientation.x = quat_rotcmd[0]
            wpose.orientation.y = quat_rotcmd[1]
            wpose.orientation.z = quat_rotcmd[2]
            wpose.orientation.w = quat_rotcmd[3]
            self.group.set_pose_target(wpose)
            self.group.go(wait=True)
            self.group.stop()
            self.group.clear_pose_targets()
        except Exception as e:
            print("Error in rotate_ee")
            print(e)
        finally:
            ee_pose = self.group.get_current_pose()
            ee_position = ee_pose.pose.position
            ee_orientation = ee_pose.pose.orientation
            return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

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
        print("GOAL")
        print(list(joint_values))
        # self.group.allow_replanning(True)
        self.group.go(list(joint_values), wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        ee_pose = self.group.get_current_pose()
        # print("EE_POSE", ee_pose)
        ee_position = ee_pose.pose.position
        ee_orientation = ee_pose.pose.orientation
        return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

    def move_to_cartesian_target(self, cartesian_values):
        """
        cartesian_values: 7 dimensional vecotr, [cartesian_position, cartesian_Quaternion]
                          [x, y, z, quat_x, quat_y, quat_z, quat_w]
        """
        print("CARTESIAN GOAL")
        print(cartesian_values)
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = cartesian_values[-4]
        pose_goal.orientation.y = cartesian_values[-3]
        pose_goal.orientation.z = cartesian_values[-2]
        pose_goal.orientation.w = cartesian_values[-1]
        pose_goal.position.x = cartesian_values[0]
        pose_goal.position.y = cartesian_values[1]
        pose_goal.position.z = cartesian_values[2]
        self.group.set_pose_target(pose_goal)

        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        
        ee_pose = self.group.get_current_pose()
        # print("EE_POSE", ee_pose)
        ee_position = ee_pose.pose.position
        ee_orientation = ee_pose.pose.orientation
        return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

    def move_to_2D_cartesian_target(self, pose, slow=False):
        try:
            waypoints = []
            next_point = self.group.get_current_pose().pose
            next_point.position.x = pose[0]
            next_point.position.y = pose[1]
            waypoints.append(copy.deepcopy(next_point))
            (plan, fraction) = self.group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0)         # jump_threshold
            if slow:
                plan = self.modify_plan(plan)
            self.group.execute(plan, wait=True) 
            self.group.stop()
            self.group.clear_pose_targets()
        except Exception as e:
            print("Error in move_to_2D_cartesian_target")
            print(e)
        finally: 
            ee_pose = self.group.get_current_pose()
            ee_position = ee_pose.pose.position
            ee_orientation = ee_pose.pose.orientation
            return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

    # linear movement planning along z axis of the reference frame
    def plan_linear_z(self, dist, slow=False):
        try:
            waypoints = []
            wpose = self.group.get_current_pose().pose
            wpose.position.z = dist
            print("move z to ", wpose.position.z)
            waypoints.append(copy.deepcopy(wpose))
            (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0)
            if slow:
                plan = self.modify_plan(plan)
            self.group.execute(plan)
            self.group.stop()
            self.group.clear_pose_targets()
        except Exception as e:
            print("Error in move_to_1D_cartesian_target")
            print(e)
        finally:
            ee_pose = self.group.get_current_pose()
            ee_position = ee_pose.pose.position
            ee_orientation = ee_pose.pose.orientation
            return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

    def stop(self):
        #try:
        self.group.stop()
        ee_pose = self.group.get_current_pose()
        # print("EE_POSE", ee_pose)
        ee_position = ee_pose.pose.position
        ee_orientation = ee_pose.pose.orientation
        return ([ee_position.x, ee_position.y, ee_position.z], [ee_orientation.x, ee_orientation.y, ee_orientation.z, ee_orientation.w])

    def pause(self):
        pass 

    def resume(self):
        pass 


    def handle_move_to_target(self, req):
        ee_pose, ee_orientation = self.move_to_joint_target(req.q)
        return TakeActionResponse(ee_pose, ee_orientation)

    def handle_move_to_cartesian_target(self, req):
        ee_pose, ee_orientation = self.move_to_cartesian_target(req.pose)
        return TakeCartesianActionResponse(ee_pose, ee_orientation)

    def handle_stop(self, req):
        ee_pose, ee_orientation = self.stop()
        return StopActionResponse(ee_pose, ee_orientation)

    def handle_move_to_2D_cartesian_target(self, req):
        ee_pose, ee_orientation = self.move_to_2D_cartesian_target(req.pose, req.slow)
        return Take2DCartesianActionResponse(ee_pose, ee_orientation)

    def handle_move_to_1D_cartesian_target(self, req):
        ee_pose, ee_orientation = self.plan_linear_z(req.z_pose, req.slow)
        return Take1DCartesianActionResponse(ee_pose, ee_orientation)

    def handle_rotate_ee(self, req):
        ee_pose, ee_orientation = self.rotate_ee(req.angle)
        return RotateEEResponse(ee_pose, ee_orientation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, help='MoveIt group name of the robot to control')
    args = parser.parse_args()

    rospy.init_node('opendr_robot_control', anonymous=True)
    
    moveit_commander.roscpp_initialize(sys.argv)
    # group_name = "panda_arm"

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(args.group_name)

    robot_arm = RobotControlNode(group)

    action_service = rospy.Service('/take_action', TakeAction, arm.handle_move_to_target)
    rospy.wait_for_service('/take_action')
    print("/take_action ready")

    cartesian_action_service = rospy.Service('/take_cartesian_action', TakeCartesianAction, arm.handle_move_to_cartesian_target)
    rospy.wait_for_service('/take_cartesian_action')
    print("/take_cartesian_action ready")

    cartesian_action_service_2d = rospy.Service('/take_2D_cartesian_action', Take2DCartesianAction, arm.handle_move_to_2D_cartesian_target)
    rospy.wait_for_service('/take_2D_cartesian_action')
    print("/take_2D_cartesian_action")

    cartesian_action_service_1d = rospy.Service('/take_1D_cartesian_action', Take1DCartesianAction, arm.handle_move_to_1D_cartesian_target)
    rospy.wait_for_service('/take_1D_cartesian_action')
    print("/take_1D_cartesian_action")

    stop_action_service = rospy.Service('/stop_actions', StopAction, arm.handle_stop)
    rospy.wait_for_service('/stop_actions')
    print("/stop_actions ready")

    rotate_ee_service = rospy.Service("/rotate_ee", RotateEE, arm.handle_rotate_ee)
    rospy.wait_for_service('/rotate_ee')
    print("/rotate_ee ready")

    rospy.spin()