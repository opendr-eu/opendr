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
from std_msgs.msg import String as ROS_String
from opendr_bridge import ROSBridge

class ContinualSlamLearner:
    def __init__(self,
                 path: Path,
                 input_image_topic : str,
                 input_distance_topic : str,
                 output_weights_topic : str,
                 fps : int = 10,
                 publish_rate : int = 20,
                 ) -> None:
        
        self.bridge = ROSBridge()
        self.delay = 1.0 / fps
        self.publish_rate = publish_rate

        self.input_image_topic = input_image_topic
        self.input_distance_topic = input_distance_topic
        self.output_weights_topic = output_weights_topic

        self.path = path
        self.learner = None


        self.do_publish = 0

        # Create caches
        self._image_cache = []
        self._distance_cache = []
        self._id_cache = []
        self._marker_position_cache = []
        self._marker_frame_id_cache = []

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

    def _init_publisher(self):
        """
        Initializing publishers.
        """
        self.output_weights_publisher = rospy.Publisher(self.output_weights_topic, ROS_String, queue_size=10)
 
    def _init_learner(self):
        """
        Creating a ContinualSLAMLearner instance with predictor and ros mode
        """
        try:
            self.learner = ContinualSLAMLearner(self.path, mode="learner", ros=True)
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

        self._cache_arriving_data(image, distance, frame_id)
        batch = self._convert_cache_into_batch()
        if len(batch) < 3:
            return
        self.learner.fit(batch)
        if self.do_publish % self.publish_rate == 0:
            message = self.learner.save()
            rospy.loginfo(f"CL-SLAM learner publishing new weights, currently in the frame {frame_id}")
            self.output_weights_publisher.publish(self.bridge.to_ros_string(message))
        self.do_publish += 1

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('opendr_continual_slam_node', anonymous=True)
        rospy.loginfo("Continual SLAM node started.")
        if self._init_learner():
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
    parser.add_argument('--output_weights_topic', type=str, default='/cl_slam/update')
    args = parser.parse_args()

    local_path = Path(__file__).parent.parent.parent.parent.parent.parent / 'src/opendr/perception/continual_slam/configs'
    path = local_path / 'singlegpu_kitti.yaml'

    node = ContinualSlamLearner(path, 
                                  args.input_image_topic,
                                  args.input_distance_topic, 
                                  args.output_weights_topic)
    node.listen()

if __name__ == '__main__':
    main()

