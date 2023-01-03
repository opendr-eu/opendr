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


import argparse
from pathlib import Path
import rospy

from opendr_bridge import ROSBridge
from opendr.perception.continual_slam.datasets.kitti import KittiDataset
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner

from sensor_msgs.msg import Image as ROS_Image, ChannelFloat32 as ROS_ChannelFloat32
from opendr_bridge import ROSBridge

class ContinualSlamPredictor:
    def __init__(self,
                 predictor: ContinualSLAMLearner,
                 input_image_topic : str,
                 input_velocity_topic : str,
                 input_distance_topic : str,
                 output_depth_topic : str,
                 output_pose_topic : str,
                 fps : int = 10,
                 ) -> None:
        
        self.bridge = ROSBridge()
        self.delay = 1.0 / fps

        self.input_image_topic = input_image_topic
        self.input_velocity_topic = input_velocity_topic
        self.input_distance_topic = input_distance_topic
        self.output_depth_topic = output_depth_topic
        self.output_pose_topic = output_pose_topic

        self.predictor = predictor

        # Create caches
        self._image_cache = []
        self._velocity_cache = []
        self._distance_cache = []
        self._id_cache = []

        self._init_subscriber()
        self._init_publisher()

    def _init_subscriber(self):
        self.input_image_subscriber = rospy.Subscriber(
            self.input_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        self.input_velocity_subscriber = rospy.Subscriber(
            self.input_velocity_topic, ROS_ChannelFloat32, self.callback, queue_size=1, buff_size=10000000)
        self.input_distance_subscriber = rospy.Subscriber(
            self.input_distance_topic, ROS_ChannelFloat32, self.callback, queue_size=1, buff_size=10000000)
    
    def _init_publisher(self):
        self.output_depth_publisher = rospy.Publisher(
            self.output_depth_topic, ROS_Image, queue_size=10)
        self.output_pose_publisher = rospy.Publisher(
            self.output_pose_topic, ROS_ChannelFloat32, queue_size=10)

    def callback(self, image: ROS_Image, velocity: ROS_ChannelFloat32, distance: ROS_ChannelFloat32):
        image = self.bridge.from_ros_image(image)
        _, velocity = self.bridge.from_ros_channel_float32(velocity)
        frame_id, distance = self.bridge.from_ros_channel_float32(distance)

        self._cache_arriving_data(image, velocity, distance, frame_id)
        batch = self._convert_cache_into_batch()

        depth, odometry = self.predictor.infer(batch)
        


    def _cache_arriving_data(self, image, velocity, distance, frame_id):
        # Cache the arriving last 3 data
        self.image_cache.append(image)
        self.velocity_cache.append(velocity)
        self.distance_cache.append(distance)
        self.id_cache.append(frame_id)

        if len(self.image_cache) > 3:
            self.image_cache.pop(0)
            self.velocity_cache.pop(0)
            self.distance_cache.pop(0)
            self.id_cache.pop(0)

    def _convert_cache_into_batch(self):
        batch = {}
        for i in range(3):
            batch[self._id_cache[i]] = (self._image_cache[i], self._velocity_cache[i], self._distance_cache[i])
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_topic', type=str, default='/opendr/dataset/image')
    parser.add_argument('--input_velocity_topic', type=str, default='/opendr/velocity')
    parser.add_argument('--input_distance_topic', type=str, default='/opendr/distance')
    parser.add_argument('--output_depth_topic', type=str, default='/opendr/predicted/image')
    parser.add_argument('--output_pose_topic', type=str, default='/opendr/predicted/pose')
    args = parser.parse_args()

    local_path = Path(__file__).parent.parent.parent / 'configs'
    predictor = ContinualSLAMLearner(local_path / 'singlegpu_kitti.yaml')

    node = ContinualSlamPredictor(predictor, 
                                  args.input_image_topic, 
                                  args.input_velocity_topic, 
                                  args.input_distance_topic, 
                                  args.output_depth_topic, 
                                  args.output_pose_topic)

