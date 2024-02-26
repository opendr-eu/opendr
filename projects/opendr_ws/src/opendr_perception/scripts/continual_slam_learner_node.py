#!/usr/bin/env python3
# Copyright 2020-2024 OpenDR European Project
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
import message_filters
import rospy
import os

from opendr_bridge import ROSBridge
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner
from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer import ReplayBuffer

from sensor_msgs.msg import Image as ROS_Image
from geometry_msgs.msg import Vector3Stamped as ROS_Vector3Stamped
from std_msgs.msg import String as ROS_String


class ContinualSlamLearner:
    def __init__(self,
                 path: str,
                 input_image_topic: str,
                 input_distance_topic: str,
                 output_weights_topic: str,
                 publish_rate: int = 20,
                 buffer_size: int = 1000,
                 save_memory: bool = True,
                 sample_size: int = 3):
        """
        Continual SLAM learner node. This node is responsible for training and updating the predictor.
        :param path: Path to the folder where the model will be saved.
        :type path: str
        :param input_image_topic: ROS topic where the input images are published.
        :type input_image_topic: str
        :param input_distance_topic: ROS topic where the input distances are published.
        :type input_distance_topic: str
        :param output_weights_topic: ROS topic where the output weights are published.
        :type output_weights_topic: str
        :param publish_rate: Rate at which the output weights are published.
        :type publish_rate: int
        :param buffer_size: Size of the replay buffer.
        :type buffer_size: int
        :param save_memory: If True, the replay buffer will be saved to memory.
        :type save_memory: bool
        :param sample_size: Number of samples (triplets from replay buffer) that will be used for training.
        :type sample_size: int
        """
        self.bridge = ROSBridge()
        self.publish_rate = publish_rate
        self.buffer_size = buffer_size
        self.save_memory = save_memory
        self.sample_size = sample_size

        self.input_image_topic = input_image_topic
        self.input_distance_topic = input_distance_topic
        self.output_weights_topic = output_weights_topic

        self.path = path
        self.learner = None
        self.sequence = None

        self.do_publish = 0

        # Create caches
        self.cache = {"image": [],
                      "distance": [],
                      "id": []}

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

        # Clear cache if new sequence is detected
        if self.sequence is None:
            self.sequence = incoming_sequence
        if self.sequence != incoming_sequence:
            self._clean_cache()
            self.sequence = incoming_sequence

        self._cache_arriving_data(image, distance, frame_id)

        # If cache is not full, return
        if len(self.cache['image']) < 3:
            return

        # Add triplet to replay buffer and sample
        item = self._convert_cache_into_triplet()
        if self.sample_size > 0:
            self.replay_buffer.add(item)
            item = ContinualSLAMLearner._input_formatter(item)
            if len(self.replay_buffer) < self.sample_size:
                return
            batch = self.replay_buffer.sample()
            # Concatenate the current triplet with the sampled batch
            batch.insert(0, item)
        else:
            item = ContinualSLAMLearner._input_formatter(item)
            batch = [item]

        # Train learner
        self.learner.fit(batch, learner=True)

        # Publish new weights
        if self.do_publish % self.publish_rate == 0:
            message = self.learner.save()
            rospy.loginfo(f"CL-SLAM learner publishing new weights, currently in the frame {frame_id}")
            ros_message = self.bridge.to_ros_string(message)
            self.output_weights_publisher.publish(ros_message)
        self.do_publish += 1

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('opendr_continual_slam_node', anonymous=True)
        if self._init_learner() and self._init_replay_buffer():
            rospy.loginfo("Continual SLAM node started.")
            self._init_publisher()
            self._init_subscribers()
            rospy.spin()

    def _init_subscribers(self):
        """
        Initializing subscribers. Here we also do synchronization between two ROS topics.
        """
        self.input_image_subscriber = message_filters.Subscriber(
            self.input_image_topic, ROS_Image, queue_size=1, buff_size=10000000)
        self.input_distance_subscriber = message_filters.Subscriber(
            self.input_distance_topic, ROS_Vector3Stamped, queue_size=1, buff_size=10000000)
        self.ts = message_filters.TimeSynchronizer([self.input_image_subscriber,
                                                    self.input_distance_subscriber], 1)
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
        env = os.getenv('OPENDR_HOME')
        path = os.path.join(env, self.path)
        print(path)
        try:
            self.learner = ContinualSLAMLearner(path, mode="learner", ros=False)
            return True
        except Exception as e:
            rospy.logerr("Continual SLAM node failed to initialize, due to learner initialization error.")
            rospy.logerr(e)
            return False

    def _init_replay_buffer(self):
        """
        Creating a replay buffer instance
        """
        env = os.getenv('OPENDR_HOME')
        path = os.path.join(env, self.path)
        try:
            self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size,
                                              save_memory=self.save_memory,
                                              dataset_config_path=path,
                                              sample_size=self.sample_size)
            return True
        except Exception:
            rospy.logerr("Continual SLAM node failed to initialize, due to replay buffer initialization error.")
            if self.sample_size > 0:
                return False
            else:
                return True

    def _clean_cache(self):
        """
        Clearing the cache.
        """
        for key in self.cache.keys():
            self.cache[key].clear()

    def _cache_arriving_data(self, image, distance, frame_id):
        """
        Caching the arriving data.
        :param image: Input image as a ROS message
        :type ROS_Image
        :param distance: Distance to the object as a ROS message
        :type ROS_Vector3Stamped
        :param frame_id: Frame id of the arriving data
        :type int
        """
        self.cache["image"].append(image)
        self.cache["distance"].append(distance)
        self.cache["id"].append(frame_id)
        if len(self.cache['image']) > 3:
            self.cache["image"].pop(0)
            self.cache["distance"].pop(0)
            self.cache["id"].pop(0)

    def _convert_cache_into_triplet(self) -> dict:
        """
        Converting the cache into a triplet.
        """
        triplet = {}
        for i in range(len(self.cache["image"])):
            triplet[self.cache['id'][i]] = (self.cache["image"][i], self.cache["distance"][i])
        return triplet


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
    parser.add_argument('-ot',
                        '--output_weights_topic',
                        type=str,
                        default='/cl_slam/update',
                        help='Output weights topic, published to Continual SLAM Predictor Node')
    parser.add_argument('-pr',
                        '--publish_rate',
                        type=int,
                        default=20,
                        help='Publish rate of the weights')
    parser.add_argument('-bs',
                        '--buffer_size',
                        type=int,
                        default=10,
                        help='Size of the replay buffer')
    parser.add_argument('-ss',
                        '--sample_size',
                        type=int,
                        default=3,
                        help='Sample size of the replay buffer. If 0 is given, only online data is used')
    parser.add_argument('-sm',
                        '--save_memory',
                        action='store_false',
                        default=True,
                        help='Whether to save memory or not. Add it to the command if you want to write to disk')
    args = parser.parse_args(rospy.myargv()[1:])

    node = ContinualSlamLearner(args.config_path,
                                args.input_image_topic,
                                args.input_distance_topic,
                                args.output_weights_topic,
                                args.publish_rate,
                                args.buffer_size,
                                args.save_memory,
                                args.sample_size)
    node.listen()


if __name__ == '__main__':
    main()
