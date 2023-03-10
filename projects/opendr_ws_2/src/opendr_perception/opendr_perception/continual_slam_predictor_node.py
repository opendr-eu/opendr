#!/usr/bin/env python3
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

import argparse
import numpy as np
from pathlib import Path
import os

from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner

import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS_Image
from geometry_msgs.msg import Vector3Stamped as ROS_Vector3Stamped
from visualization_msgs.msg import MarkerArray as ROS_MarkerArray
from std_msgs.msg import String as ROS_String
from opendr_bridge import ROS2Bridge

class ContinualSlamPredictor(Node):
    def __init__(self,
                 path: Path,
                 input_image_topic : str,
                 input_distance_topic : str,
                 output_depth_topic : str,
                 output_pose_topic : str,
                 update_topic : str):
        super().__init__("opendr_continual_slam_predictor_node")
        """
        Continual SLAM predictor node. This node is responsible for publishing predicted pose and depth outputs.
        :param path: Path to the folder where the model will be saved.
        :type path: str
        :param input_image_topic: ROS topic where the input image is published.
        :type input_image_topic: str
        :param input_distance_topic: ROS topic where the input distance is published.
        :type input_distance_topic: str
        :param output_depth_topic: ROS topic where the output depth will be published.
        :type output_depth_topic: str
        :param output_pose_topic: ROS topic where the output pose will be published.
        :type output_pose_topic: str
        :param update_topic: ROS topic where the update signal is published.
        :type update_topic: str
        :param fps: Frequency of the node in Hz.
        :type fps: int
        """
        self.bridge = ROS2Bridge()

        self.input_image_topic = input_image_topic
        self.input_distance_topic = input_distance_topic
        self.output_depth_topic = output_depth_topic
        self.output_pose_topic = output_pose_topic
        self.update_topic = update_topic

        self.path = path
        self.predictor = None
        self.sequence = None
        self.color = None
        self.frame_id = None

        # Create caches
        self.cache = {
            "image": [],
            "distance": [],
            "id": [],
            "marker_position": [],
            "marker_frame_id": []}

    def _init_subscribers(self):
        """
        Initializing subscribers. Here we also do synchronization between two ROS topics.
        """
        self.input_image_subscriber = message_filters.Subscriber(
            self, ROS_Image, self.input_image_topic)
        self.input_distance_subscriber = message_filters.Subscriber(
            self, ROS_Vector3Stamped, self.input_distance_topic)
        self.ts = message_filters.TimeSynchronizer([self.input_image_subscriber,
                                                    self.input_distance_subscriber], 10)
        self.ts.registerCallback(self.callback)

        self.update_subscriber = self.create_subscription(ROS_String, self.update_topic, self.update, 1)

    def _init_publisher(self):
        """
        Initializing publishers.
        """
        self.output_depth_publisher = self.create_publisher(
            ROS_Image, self.output_depth_topic, 10)
        self.output_pose_publisher = self.create_publisher(
            ROS_MarkerArray, self.output_pose_topic, 10)

    def _init_predictor(self):
        """
        Creating a ContinualSLAMLearner instance with predictor and ros mode
        """
        env = os.getenv('OPENDR_HOME')
        path = os.path.join(env, self.path)
        try:
            self.predictor = ContinualSLAMLearner(path, mode="predictor", ros=False, do_loop_closure=True)
            return True
        except Exception as e:
            self.get_logger().error("Continual SLAM node failed to initialize, due to predictor initialization error.")
            self.get_logger().error(e)
            return False
                
    def _clean_cache(self):
        """
        Cleaning the cache
        """
        for key in self.cache.keys():
            self.cache[key].clear()

    def _cache_arriving_data(self, image, distance, frame_id):
        """
        Caching arriving data
        :param image: Input image
        :type image: np.ndarray
        :param distance: Distance to the object
        :type distance: float
        :param frame_id: Frame id
        :type frame_id: str
        """
        # Cache the arriving last 3 data
        self.cache['image'].append(image)
        self.cache['distance'].append(distance)
        self.cache['id'].append(frame_id)
        if len(self.cache['image']) > 3:
            self.cache['image'].pop(0)
            self.cache['distance'].pop(0)
            self.cache['id'].pop(0)

    def _convert_cache_into_triplet(self) -> dict:
        triplet = {}
        for i in range(len(self.cache['image'])):
            triplet[self.cache['id'][i]] = (self.cache['image'][i], self.cache['distance'][i])
        return triplet

    def callback(self, image: ROS_Image, distance: ROS_Vector3Stamped):
        """
        Callback method of predictor node.
        :param image: Input image as a ROS message
        :type ROS_Image
        :param distance: Distance to the object as a ROS message
        :type ROS_Vector3Stamped
        """
        stamp = self.get_clock().now().to_msg()
        image = self.bridge.from_ros_image(image)
        frame_id, distance = self.bridge.from_ros_vector3_stamped(distance)
        self.frame_id = frame_id
        incoming_sequence = frame_id.split("_")[0]
        distance = distance[0]

        # If new sequence is detected, clean the cache
        if self.sequence is None:
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))
        if self.sequence != incoming_sequence:
            self._clean_cache()
            self.predictor.step = 0
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))

        # Cache incoming data
        self._cache_arriving_data(image, distance, frame_id)
        triplet = self._convert_cache_into_triplet()
        if len(triplet) < 3:
            return

        # Infer depth and pose
        depth, _, _, lc, pose_graph = self.predictor.infer(triplet)
        if not lc:
            points = pose_graph.return_last_positions(n=10)
            if not len(points):
                return
            for point in points:
                position = [-point[0], 0.0, -point[2]]

                self.cache["marker_position"].append(position)
                self.cache["marker_frame_id"].append("map")
            if self.color is None:
                self.color = [255, 0, 0]
            rgba = (self.color[0], self.color[1], self.color[2], 1.0)
        else:
            self.cache["marker_position"].clear()
            self.cache["marker_frame_id"].clear()
            points = pose_graph.return_all_positions()
            for point in points:
                position = [-point[0], 0.0, -point[2]]
                self.cache["marker_position"].append(position)
                self.cache["marker_frame_id"].append("map")

        rgba = (self.color[0], self.color[1], self.color[2], 1.0)
        marker_list = self.bridge.to_ros_marker_array(self.cache['marker_position'],
                                                self.cache['marker_frame_id'],
                                                stamp,
                                                rgba)

        depth = self.bridge.to_ros_image(depth)

        # self.get_logger().info(f"CL-SLAM predictor is currently predicting depth and pose. Current frame id {frame_id}")
        self.output_depth_publisher.publish(depth)
        self.output_pose_publisher.publish(marker_list)
            

    def update(self, message: ROS_String):
        """
        Update the predictor with the new data
        :param message: ROS message
        :type ROS_Byte
        """
        message = self.bridge.from_ros_string(message)
        self.get_logger().info(f"CL-SLAM predictor is currently updating its weights from {message}.\n"
                               f"Last predicted frame id is {self.frame_id}\n")
        self.predictor.load(weights_folder=message)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        if self._init_predictor():
            self._init_publisher()
            self._init_subscribers()
            self.get_logger().info("Continual SLAM node started.")
            rclpy.spin(self)

            # Destroy the node explicitly
            # (optional - otherwise it will be done automatically
            # when the garbage collector destroys the node object)
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default='src/opendr/perception/continual_slam/configs/singlegpu_kitti.yaml',
                        help='Path to the config file')
    parser.add_argument('-it',
                        '--input_image_topic',
                        type=str,
                        default='/cl_slam/image',
                        help='Input image topic, listened from Continual SLAM Dataset Node')
    parser.add_argument('-dt',
                        '--input_distance_topic',
                        type=str,
                        default='/cl_slam/distance',
                        help='Input distance topic, listened from Continual SLAM Dataset Node')
    parser.add_argument('-odt',
                        '--output_depth_topic',
                        type=str,
                        default='/opendr/predicted/image',
                        help='Output depth topic, published to Continual SLAM Dataset Node')
    parser.add_argument('-opt',
                        '--output_pose_topic',
                        type=str,
                        default='/opendr/predicted/pose',
                        help='Output pose topic, published to Continual SLAM Dataset Node')
    parser.add_argument('-ut',
                        '--update_topic',
                        type=str,
                        default='/cl_slam/update',
                        help='Update topic, listened from Continual SLAM Dataset Node')
    args = parser.parse_args()

    node = ContinualSlamPredictor(args.config_path,
                                  args.input_image_topic,
                                  args.input_distance_topic,
                                  args.output_depth_topic,
                                  args.output_pose_topic,
                                  args.update_topic)
    node.listen()


if __name__ == '__main__':
    main()
