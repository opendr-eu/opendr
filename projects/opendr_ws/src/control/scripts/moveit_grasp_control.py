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

import rospy 


class RobotControlNode(object):

    def __init__(self, group="panda_arm"):

        self.group = group
        self.group.set_max_velocity_scaling_factor(0.2)

        self.rotate_ee_service = rospy.Service("/rotate_ee", RotateEE, self.handle_rotate_ee)
        self.stop_action_service = rospy.Service('/stop_actions', StopAction, self.handle_stop)
        self.action_service = rospy.Service('/take_action', TakeAction, self.handle_move_to_target)
        self.cartesian_action_service = rospy.Service('/take_cartesian_action', TakeCartesianAction, self.handle_move_to_cartesian_target)
        self.cartesian_action_service_2d = rospy.Service('/take_2D_cartesian_action', Take2DCartesianAction, self.handle_move_to_2D_cartesian_target)
        self.cartesian_action_service_1d = rospy.Service('/take_1D_cartesian_action', Take1DCartesianAction, self.handle_move_to_1D_cartesian_target)

    def rotate_ee(self, angle):  
        wpose = self.group.get_current_pose().pose
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w])
        yaw = angle + math.pi/4 
        if yaw > math.pi/4:
            yaw = yaw - math.pi
        quat_rotcmd = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        wpose.orientation.x = quat_rotcmd[0]
        wpose.orientation.y = quat_rotcmd[1]
        wpose.orientation.z = quat_rotcmd[2]
        wpose.orientation.w = quat_rotcmd[3]
        self.group.set_pose_target(wpose)
        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

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
        self.group.stop()
        self.group.clear_pose_targets()

    def move_to_cartesian_target(self, cartesian_values):
        """
        cartesian_values: 7 dimensional vecotr, [cartesian_position, cartesian_Quaternion]
                          [x, y, z, quat_x, quat_y, quat_z, quat_w]
        """
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


    def move_to_2D_cartesian_target(self, pose, slow=False):
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

    def plan_linear_z(self, dist, slow=False):
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

    def stop(self):
        self.group.stop()
        ee_pose = self.group.get_current_pose()

    def pause(self):
        pass 

    def resume(self):
        pass 

    def handle_move_to_target(self, req):
        self.move_to_joint_target(req.q)
        return TakeActionResponse()

    def handle_move_to_cartesian_target(self, req):
        self.move_to_cartesian_target(req.pose)
        return TakeCartesianActionResponse()

    def handle_stop(self, req):
        self.stop()
        return StopActionResponse()

    def handle_move_to_2D_cartesian_target(self, req):
        self.move_to_2D_cartesian_target(req.pose, req.slow)
        return Take2DCartesianActionResponse()

    def handle_move_to_1D_cartesian_target(self, req):
        self.plan_linear_z(req.z_pose, req.slow)
        return Take1DCartesianActionResponse()

    def handle_rotate_ee(self, req):
        self.rotate_ee(req.angle)
        return RotateEEResponse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, help='MoveIt group name of the robot to control')
    args = parser.parse_args()

    rospy.init_node('opendr_robot_control', anonymous=True)
    
    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(args.group_name)

    robot_arm = RobotControlNode(group)

    rospy.spin()