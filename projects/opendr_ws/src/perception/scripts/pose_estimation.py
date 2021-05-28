#!/usr/bin/env python
# Copyright 2020-2021 OpenDR European Project
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
import torch
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import draw
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner


class PoseEstimationNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", output_image_topic="/opendr/image_pose_annotated",
                 pose_annotations_topic="/opendr/poses", device="cuda"):
        """
        Creates a ROS Node for pose detection
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param pose_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated pose annotations)
        :type pose_annotations_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        if output_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        else:
            self.image_publisher = None

        if pose_annotations_topic is not None:
            self.pose_publisher = rospy.Publisher(pose_annotations_topic, Detection2DArray, queue_size=10)
        else:
            self.pose_publisher = None

        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

        self.bridge = ROSBridge()

        # Initialize the pose estimation
        self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=0,
                                                         mobilenet_use_stride=False,
                                                         half_precision=False)
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_pose_estimation', anonymous=True)
        rospy.loginfo("Pose estimation node started!")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data)

        # Run pose estimation
        poses = self.pose_estimator.infer(image)

        # Get an OpenCV image back
        image = np.float32(image.numpy())
        #  Annotate image and publish results
        for pose in poses:
            if self.pose_publisher is not None:
                ros_pose = self.bridge.to_ros_pose(pose)
                self.pose_publisher.publish(ros_pose)
                # We get can the data back using self.bridge.from_ros_pose(ros_pose)
                # e.g., opendr_pose = self.bridge.from_ros_pose(ros_pose)
                draw(image, pose)

        if self.image_publisher is not None:
            message = self.bridge.to_ros_image(np.uint8(image))
            self.image_publisher.publish(message)


if __name__ == '__main__':
    # Select the device for running the
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'

    pose_estimation_node = PoseEstimationNode(device=device)
    pose_estimation_node.listen()
