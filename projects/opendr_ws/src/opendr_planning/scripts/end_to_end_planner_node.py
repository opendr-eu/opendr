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
import numpy as np
import webots_ros.srv
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped, PointStamped
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_from_quaternion
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_to_quaternion


class EndToEndPlannerNode:

    def __init__(self):
        """
        Creates a ROS Node for end-to-end planner
        """
        self.node_name = "opendr_end_to_end_planner"
        self.bridge = CvBridge()
        self.model_name = ""
        self.current_pose = PoseStamped()
        self.target_pose = PoseStamped()
        self.current_pose.header.frame_id = "map"
        self.target_pose.header.frame_id = "map"
        rospy.init_node(self.node_name, anonymous=True)
        self.r = rospy.Rate(25)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        counter = 0
        while self.model_name == "":
            self.r.sleep()
            counter += 1
            if counter > 25:
                break
        if self.model_name == "":
            rospy.loginfo("Webots model is not started!")
            return
        self.input_depth_image_topic = "/range_finder/range_image"
        self.position_topic = "/gps/values"
        self.orientation_topic = "/inertial_unit/quaternion"
        self.ros_srv_range_sensor_enable = rospy.ServiceProxy(
            "/range_finder/enable", webots_ros.srv.set_int)
        self.ros_srv_gps_sensor_enable = rospy.ServiceProxy(
            "/gps/enable", webots_ros.srv.set_int)
        self.ros_srv_inertial_unit_enable = rospy.ServiceProxy(
            "/inertial_unit/enable", webots_ros.srv.set_int)
        self.end_to_end_planner = EndToEndPlanningRLLearner(env=None)

        try:
            self.ros_srv_gps_sensor_enable(1)
            self.ros_srv_inertial_unit_enable(1)
            self.ros_srv_range_sensor_enable(1)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        self.ros_pub_current_pose = rospy.Publisher('current_uav_pose', PoseStamped, queue_size=10)
        self.ros_pub_target_pose = rospy.Publisher('target_uav_pose', PoseStamped, queue_size=10)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.Subscriber(self.orientation_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.position_topic, PointStamped, self.gps_callback)
        rospy.Subscriber(self.input_depth_image_topic, Image, self.range_callback, queue_size=1)
        rospy.spin()

    def range_callback(self, data):
        image_arr = self.bridge.imgmsg_to_cv2(data)
        self.range_image = ((np.clip(image_arr.reshape((64, 64, 1)), 0, 15) / 15.) * 255).astype(np.uint8)
        observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.array([5, 0, 0])}
        action = self.end_to_end_planner.infer(observation, deterministic=True)[0]
        self.publish_poses(action)

    def gps_callback(self, data):  # for no dynamics
        self.current_pose.header.stamp = rospy.Time.now()
        self.current_pose.pose.position.x = -data.point.x
        self.current_pose.pose.position.y = -data.point.y
        self.current_pose.pose.position.z = data.point.z

    def imu_callback(self, data):  # for no dynamics
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
        self.target_pose.header.stamp = rospy.Time.now()
        self.target_pose.pose.position.x = self.current_pose.pose.position.x + forward_step * np.cos(
            self.current_yaw) - side_step * np.sin(self.current_yaw)
        self.target_pose.pose.position.y = self.current_pose.pose.position.y + forward_step * np.sin(
            self.current_yaw) + side_step * np.cos(self.current_yaw)
        self.target_pose.pose.position.z = self.current_pose.pose.position.z
        self.target_pose.pose.orientation = euler_to_quaternion(0, 0, yaw=self.current_yaw+yaw_step)
        self.ros_pub_target_pose.publish(self.target_pose)


if __name__ == '__main__':
    end_to_end_planner_node = EndToEndPlannerNode()
    end_to_end_planner_node.listen()
