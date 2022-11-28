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

import rclpy
from rclpy.node import Node
import numpy as np
import webots_ros.srv
# import ros_numpy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped, PointStamped
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_from_quaternion
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_to_quaternion


class EndToEndPlannerNode(Node):

    def __init__(self):
        """
        Creates a ROS Node for end-to-end planner
        """
        super().__init__("opendr_end_to_end_planner_node")
        self.model_name = ""
        self.current_pose = PoseStamped()
        self.target_pose = PoseStamped()
        self.current_pose.header.frame_id = "map"
        self.target_pose.header.frame_id = "map"
        # self.r = rclpy.Rate(25)
        # self.create_subscription("/model_name", String, self.model_name_callback)
        # counter = 0
        # while self.model_name == "":
        #     self.r.sleep()
        #     counter += 1
        #     if counter > 25:
        #         break
        # if self.model_name == "":
        #     rospy.loginfo("Webots model is not started!")
        #     return
        self.input_depth_image_topic = "/quad_plus_sitl/range_finder"
        self.position_topic = "/quad_plus_sitl/gps1"
        self.orientation_topic = "/imu"
        # self.ros_srv_range_sensor_enable = rospy.ServiceProxy(
        #     "/range_finder/enable", webots_ros.srv.set_int)
        # self.ros_srv_gps1_sensor_enable = rospy.ServiceProxy(
        #     "/gps1/enable", webots_ros.srv.set_int)
        # self.ros_srv_inertial_unit_enable = rospy.ServiceProxy(
        #     "/inertial_unit/enable", webots_ros.srv.set_int)
        self.end_to_end_planner = EndToEndPlanningRLLearner(env=None)

        # try:
        #     self.ros_srv_gps1_sensor_enable(1)
        #     self.ros_srv_inertial_unit_enable(1)
        #     self.ros_srv_range_sensor_enable(1)
        # except rospy.ServiceException as exc:
        #     print("Service did not process request: " + str(exc))
        self.ros2_pub_current_pose = self.create_publisher(PoseStamped, 'current_uav_pose', queue_size=10)
        self.ros2_pub_target_pose = self.create_publisher(PoseStamped, 'target_uav_pose', queue_size=10)
        self.create_subscription(Imu, self.orientation_topic, self.imu_callback)
        self.create_subscription(PointStamped, self.position_topic, self.gps_callback)
        self.create_subscription(Image, self.input_depth_image_topic, self.range_callback, queue_size=1)
        self.get_logger().info("End-to-end planning node initialized.")

    def range_callback(self, data):
        print(data)
        # image_arr = ros_numpy.numpify(data)
        # self.range_image = ((np.clip(image_arr.reshape((64, 64, 1)), 0, 15) / 15.) * 255).astype(np.uint8)
        # observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.array([5, 0, 0])}
        # action = self.end_to_end_planner.infer(observation, deterministic=True)[0]
        # self.publish_poses(action)

    def gps_callback(self, data):
        self.current_pose.header.stamp = self.get_clock().now()
        self.current_pose.pose.position.x = -data.point.x
        self.current_pose.pose.position.y = -data.point.y
        self.current_pose.pose.position.z = data.point.z

    def imu_callback(self, data):
        self.current_orientation = data.orientation
        self.current_yaw = euler_from_quaternion(data.orientation)["yaw"]
        self.current_pose.pose.orientation = euler_to_quaternion(0, 0, yaw=self.current_yaw)

    def model_name_callback(self, data):
        if data.data[:5] == "robot":
            self.model_name = data.data
        if data.data[:4] == "quad":
            self.model_name = data.data

    def publish_poses(self, action):
        self.ros_pub_current_pose.publish(self.current_pose)
        forward_step = np.cos(action[0] * 22.5 / 180 * np.pi)
        side_step = np.sin(action[0] * 22.5 / 180 * np.pi)
        yaw_step = action[1] * 22.5 / 180 * np.pi
        self.target_pose.header.stamp = self.get_clock().now()
        self.target_pose.pose.position.x = self.current_pose.pose.position.x + forward_step * np.cos(
            self.current_yaw) - side_step * np.sin(self.current_yaw)
        self.target_pose.pose.position.y = self.current_pose.pose.position.y + forward_step * np.sin(
            self.current_yaw) + side_step * np.cos(self.current_yaw)
        self.target_pose.pose.position.z = self.current_pose.pose.position.z
        self.target_pose.pose.orientation = euler_to_quaternion(0, 0, yaw=self.current_yaw+yaw_step)
        self.ros_pub_target_pose.publish(self.target_pose)


def main(args=None):
    rclpy.init(args=args)
    end_to_end_planner_node = EndToEndPlannerNode()
    rclpy.spin(end_to_end_planner_node)
    end_to_end_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
