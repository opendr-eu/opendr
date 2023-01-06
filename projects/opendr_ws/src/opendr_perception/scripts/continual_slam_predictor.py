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
import time
from pathlib import Path
import message_filters
import rospy

from opendr_bridge import ROSBridge
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner

from sensor_msgs.msg import Image as ROS_Image, ChannelFloat32 as ROS_ChannelFloat32
from visualization_msgs.msg import Marker as ROS_Marker, MarkerArray as ROS_MarkerArray
from opendr_bridge import ROSBridge

class ContinualSlamPredictor:
    def __init__(self,
                 path: Path,
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

        self.path = path
        self.predictor = None

        # Create caches
        self._image_cache = []
        self._velocity_cache = []
        self._distance_cache = []
        self._id_cache = []
        self._marker_position_cache = []
        self._marker_frame_id_cache = []

    def _init_subscribers(self):
        self.input_image_subscriber = message_filters.Subscriber(
            self.input_image_topic, ROS_Image, queue_size=1, buff_size=10000000)
        self.input_velocity_subscriber = message_filters.Subscriber(
            self.input_velocity_topic, ROS_ChannelFloat32, queue_size=1, buff_size=10000000)
        self.input_distance_subscriber = message_filters.Subscriber(
            self.input_distance_topic, ROS_ChannelFloat32, queue_size=1, buff_size=10000000)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.input_image_subscriber, 
                                               self.input_velocity_subscriber, 
                                               self.input_distance_subscriber], 1, 1, allow_headerless=True)
    
    def _init_publisher(self):
        self.output_depth_publisher = rospy.Publisher(
            self.output_depth_topic, ROS_Image, queue_size=10)
        self.output_pose_publisher = rospy.Publisher(
            self.output_pose_topic, ROS_MarkerArray, queue_size=10)

    def _init_predictor(self):
        try:
            self.predictor = ContinualSLAMLearner(self.path)
            return True
        except Exception as e:
            rospy.logerr("Continual SLAM node failed to initialize, due to predictor initialization error.")
            rospy.logerr(e)
            return False

    def callback(self, image: ROS_Image, velocity: ROS_ChannelFloat32, distance: ROS_ChannelFloat32):
        print("I reached here 3")
        print(type(image))
        image = self.bridge.from_ros_image(image)
        _, velocity = self.bridge.from_ros_channel_float32(velocity)
        frame_id, distance = self.bridge.from_ros_channel_float32(distance)

        self._cache_arriving_data(image, velocity, distance, frame_id)
        batch = self._convert_cache_into_batch()

        depth, odometry = self.predictor.infer(batch)
        translation = odometry[0]
        x = translation[:, -1][1]
        y = translation[:, -1][2]
        z = translation[:, -1][0]

        position = [x, y, z]
        frame_id = self._id_cache[1]
        self._marker_position_cache.append(position)
        self._marker_frame_id_cache.append(frame_id)

        marker_list = self.bridge.to_ros_marker_array(self._marker_position_cache, self._marker_frame_id_cache)
        depth = self.bridge.to_ros_image(depth)

        self.output_depth_publisher.publish(depth)
        self.output_pose_publisher.publish(marker_list)

        time.sleep(self.delay)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('opendr_continual_slam_node', anonymous=True)
        rospy.loginfo("Continual SLAM node started.")
        if self._init_predictor():
            self._init_publisher()
            self._init_subscribers()
            print("I reached here")
            self.ts.registerCallback(self.callback)
            print("I reached here 2")
            rospy.spin()

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

    local_path = Path(__file__).parent.parent.parent.parent.parent.parent / 'src/opendr/perception/continual_slam/configs'
    print(local_path)
    path = local_path / 'singlegpu_kitti.yaml'

    node = ContinualSlamPredictor(path, 
                                  args.input_image_topic, 
                                  args.input_velocity_topic, 
                                  args.input_distance_topic, 
                                  args.output_depth_topic, 
                                  args.output_pose_topic)
    node.listen()

if __name__ == '__main__':
    main()

