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
                 input_image_topic : str,
                 input_distance_topic : str,
                 output_depth_topic : str,
                 output_pose_topic : str,
                 update_topic : str,
                 fps : int = 10):
        
        self.bridge = ROSBridge()
        self.delay = 1.0 / fps

        self.input_image_topic = input_image_topic
        self.input_distance_topic = input_distance_topic
        self.output_depth_topic = output_depth_topic
        self.output_pose_topic = output_pose_topic
        self.update_topic = update_topic

        self.path = path
        self.predictor = None
        self.sequence = None
        self.color = None

        # Create caches
        self.cache = {
            "image": [],
            "distance": [],
            "id": [],
            "marker_position": [],
            "marker_frame_id": []}
        self.odometry = None

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
        try:
            self.predictor = ContinualSLAMLearner(self.path, mode="predictor", ros=True)
            return True
        except Exception as e:
            rospy.logerr("Continual SLAM node failed to initialize, due to predictor initialization error.")
            rospy.logerr(e)
            return False
                
    def _clean_cache(self):
        """
        Cleaning the cache
        """
        for key in self.cache.keys():
            self.cache[key].clear()
        self.odometry = None

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
        for i in range(len(self._image_cache)):
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
        image = self.bridge.from_ros_image(image)
        frame_id, distance = self.bridge.from_ros_vector3_stamped(distance)
        incoming_sequence = frame_id.split("_")[0]
        distance = distance[0]

        # If new sequence is detected, clean the cache
        if self.sequence is None:
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))
        if self.sequence != incoming_sequence:
            self._clean_cache()
            self.sequence = incoming_sequence
            self.color = list(np.random.choice(range(256), size=3))

        # Cache incoming data
        self._cache_arriving_data(image, distance, frame_id)
        triplet = self._convert_cache_into_triplet()
        if len(triplet) < 3:
            return

        # Infer depth and pose
        depth, new_odometry = self.predictor.infer(triplet)
        if self.odometry is None:
            self.odometry = new_odometry
        else:
            self.odometry = self.odometry @ new_odometry
        translation = self.odometry[0][:3, 3]
        position = [-translation[0], 0, -translation[2]]

        self.cache["marker_position"].append(position)
        self.cache["marker_frame_id"].append("map")
        if self.color is None:
            self.color = [255, 0, 0]
        rgba = (self.color[0], self.color[1], self.color[2], 1.0)
        marker_list = self.bridge.to_ros_marker_array(self.cache['marker_position'], self.cache['marker_frame_id'], rgba)
        depth = self.bridge.to_ros_image(depth)

        rospy.loginfo(f"CL-SLAM predictor is currently predicting depth and pose. Current frame id {frame_id}")
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
        self.predictor.load(message=message)

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
