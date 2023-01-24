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
import numpy as np
import time
from pathlib import Path
import message_filters
import rospy

from opendr_bridge import ROSBridge
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner

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
                 fps : int = 10,
                 ) -> None:
        
        self.bridge = ROSBridge()
        self.delay = 1.0 / fps

        self.input_image_topic = input_image_topic
        self.input_distance_topic = input_distance_topic
        self.output_depth_topic = output_depth_topic
        self.output_pose_topic = output_pose_topic
        self.update_topic = update_topic

        self.path = path
        self.predictor = None

        # Create caches
        self._image_cache = []
        self._distance_cache = []
        self._id_cache = []
        self._marker_position_cache = []
        self._marker_frame_id_cache = []
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
            self.predictor = ContinualSLAMLearner(self.path, mode="predictor", ros=False)
            return True
        except Exception as e:
            rospy.logerr("Continual SLAM node failed to initialize, due to predictor initialization error.")
            rospy.logerr(e)
            return False

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
        distance = distance[0]
        temp = frame_id
        self._cache_arriving_data(image, distance, frame_id)
        batch = self._convert_cache_into_batch()
        if len(batch) < 3:
            return
        depth, new_odometry = self.predictor.infer(batch)
        if self.odometry is None:
            self.odometry = new_odometry
        else:
            self.odometry = self.odometry @ new_odometry
        translation = self.odometry[0][:3, 3]
        # print(translation)
        # print(self.odometry)
        # print(translation.shape)
        x = translation[0]
        #y = translation[:, -1][1]
        y = 0
        z = translation[2]
        position = [x, y, z]

        frame_id = "map"
        self._marker_position_cache.append(position)
        self._marker_frame_id_cache.append(frame_id)

        marker_list = self.bridge.to_ros_marker_array(self._marker_position_cache, self._marker_frame_id_cache)
        depth = self.bridge.to_ros_image(depth)

        rospy.loginfo(f"CL-SLAM predictor is currently predicting depth and pose. Current frame id {temp}")
        self.output_depth_publisher.publish(depth)
        self.output_pose_publisher.publish(marker_list)

        # time.sleep(self.delay)

    def update(self, message: ROS_String):
        """
        Update the predictor with the new data
        :param message: ROS message
        :type ROS_Byte
        """
        rospy.loginfo("CL-SLAM predictor is currently updating its weights.")
        message = self.bridge.from_ros_string(message)
        self.predictor.load(message)

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
            rospy.spin()

    def _cache_arriving_data(self, image, distance, frame_id):
        # Cache the arriving last 3 data
        self._image_cache.append(image)
        self._distance_cache.append(distance)
        self._id_cache.append(frame_id)

        if len(self._image_cache) > 3:
            self._image_cache.pop(0)
            self._distance_cache.pop(0)
            self._id_cache.pop(0)

    def _convert_cache_into_batch(self):
        batch = {}
        for i in range(len(self._image_cache)):
            batch[self._id_cache[i]] = (self._image_cache[i], self._distance_cache[i])
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_topic', type=str, default='/cl_slam/image')
    parser.add_argument('--input_distance_topic', type=str, default='/cl_slam/distance')
    parser.add_argument('--output_depth_topic', type=str, default='/opendr/predicted/image')
    parser.add_argument('--output_pose_topic', type=str, default='/opendr/predicted/pose')
    parser.add_argument('--update_topic', type=str, default='/cl_slam/update')
    args = parser.parse_args()

    local_path = Path(__file__).parent.parent.parent.parent.parent.parent / 'src/opendr/perception/continual_slam/configs'
    path = local_path / 'singlegpu_kitti.yaml'

    node = ContinualSlamPredictor(path, 
                                  args.input_image_topic,
                                  args.input_distance_topic, 
                                  args.output_depth_topic, 
                                  args.output_pose_topic,
                                  args.update_topic)
    node.listen()

if __name__ == '__main__':
    main()

