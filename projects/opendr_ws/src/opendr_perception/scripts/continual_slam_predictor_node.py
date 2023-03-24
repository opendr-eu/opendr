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
import rospy
from sensor_msgs.msg import Image as ROS_Image
from geometry_msgs.msg import Vector3Stamped as ROS_Vector3Stamped
from visualization_msgs.msg import MarkerArray as ROS_MarkerArray
from std_msgs.msg import String as ROS_String
from opendr_bridge import ROSBridge


class ContinualSlamPredictor:
    def __init__(self,
                 path: Path,
                 input_image_topic: str,
                 input_distance_topic: str,
                 output_depth_topic: str,
                 output_pose_topic: str,
                 update_topic: str):
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
        """
        self.bridge = ROSBridge()

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

    def callback(self, image: ROS_Image, distance: ROS_Vector3Stamped):
        """
        Callback method of predictor node.
        :param image: Input image as a ROS message
        :type ROS_Image
        :param distance: Distance to the object as a ROS message
        :type ROS_Vector3Stamped
        """
        # Data-preprocessing
        image = self.bridge.from_ros_image(image)
        frame_id, distance = self.bridge.from_ros_vector3_stamped(distance)
        self.frame_id = frame_id
        incoming_sequence = frame_id.split("_")[0]
        distance = distance[0]

        self._check_sequence(incoming_sequence)
        self._cache_arriving_data(image, distance, frame_id)
        triplet = self._convert_cache_into_triplet()
        # If triplet is not ready, return
        if len(triplet) < 3:
            return
        # Infer depth and pose
        depth = self._infer(triplet)
        if depth is None:
            return
        rgba = (self.color[0], self.color[1], self.color[2], 1.0)
        marker_list = self.bridge.to_ros_marker_array(self.cache['marker_position'],
                                                      self.cache['marker_frame_id'],
                                                      rgba)

        depth = self.bridge.to_ros_image(depth)
        self.output_depth_publisher.publish(depth)
        self.output_pose_publisher.publish(marker_list)

    def update(self, message: ROS_String):
        """
        Update the predictor with the new data
        :param message: ROS message
        :type ROS_Byte
        """
        rospy.loginfo("CL-SLAM predictor is currently updating its weights.")
        message = self.bridge.from_ros_string(message)
        self.predictor.load(weights_folder=message)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('opendr_continual_slam_node', anonymous=True)
        if self._init_predictor():
            rospy.loginfo("Continual SLAM node started.")
            self._init_publisher()
            self._init_subscribers()
            rospy.spin()

    # Auxiliary functions
    def _init_subscribers(self):
        """
        Initializing subscribers. Here we also do synchronization between two ROS topics.
        """
        self.input_image_subscriber = message_filters.Subscriber(
            self.input_image_topic, ROS_Image, queue_size=1, buff_size=10000000)
        self.input_distance_subscriber = message_filters.Subscriber(
            self.input_distance_topic, ROS_Vector3Stamped, queue_size=1, buff_size=10000000)
        self.ts = message_filters.TimeSynchronizer([self.input_image_subscriber, self.input_distance_subscriber], 1)
        self.ts.registerCallback(self.callback)

        self.update_subscriber = rospy.Subscriber(self.update_topic, ROS_String, self.update, queue_size=1, buff_size=10000000)

    def _init_publisher(self):
        """
        Initializing publishers.
        """
        self.output_depth_publisher = rospy.Publisher(
            self.output_depth_topic, ROS_Image, queue_size=10)
        self.output_pose_publisher = rospy.Publisher(
            self.output_pose_topic, ROS_MarkerArray, queue_size=10)

    def _init_predictor(self):
        """
        Creating a ContinualSLAMLearner instance with predictor and ros mode
        """
        env = os.getenv('OPENDR_HOME')
        path = os.path.join(env, self.path)
        try:
            self.predictor = ContinualSLAMLearner(path, mode="predictor", ros=False, do_loop_closure=True)
            return True
        except Exception:
            rospy.logerr("Continual SLAM node failed to initialize, due to predictor initialization error.")
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
        """
        Converting the cached data into a triplet dictionary
        return: Triplet dictionary
        rtype: dict
        """
        triplet = {}
        for i in range(len(self.cache['image'])):
            triplet[self.cache['id'][i]] = (self.cache['image'][i], self.cache['distance'][i])
        return triplet

    def _check_sequence(self, incoming_sequence):
        """
        Checking if the new sequence is detected
        :param incoming_sequence: Incoming sequence
        :type incoming_sequence: str
        """
        # If new sequence is detected, clean the cache
        if self.sequence is None:
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))
        if self.sequence != incoming_sequence:
            self._clean_cache()
            self.predictor.step = 0
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))

    def _infer(self, triplet: dict):
        """
        Infer the triplet
        :param triplet: Triplet
        :type triplet: dict
        :return: Depth image
        :rtype: Image
        """
        depth, _, _, lc, pose_graph = self.predictor.infer(triplet)
        if not lc:
            points = pose_graph.return_last_positions(n=5)
            if not len(points):
                return None
            for point in points:
                position = [-point[0], 0.0, -point[2]]

                self.cache["marker_position"].append(position)
                self.cache["marker_frame_id"].append("map")
            if self.color is None:
                self.color = [255, 0, 0]
        else:
            self.cache["marker_position"].clear()
            self.cache["marker_frame_id"].clear()
            points = pose_graph.return_all_positions()
            for point in points:
                position = [-point[0], 0.0, -point[2]]
                self.cache["marker_position"].append(position)
                self.cache["marker_frame_id"].append("map")
        return depth


def main():
    parser = argparse.ArgumentParser()

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
