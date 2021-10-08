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
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from simulation.srv import Mesh_vc
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import draw
from opendr.simulation.human_model_generation.pifu_generator_learner import PIFuGeneratorLearner
import cv2
import os
from cv_bridge import CvBridge
from std_msgs.msg import Bool

class PifuNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", output_human_mopdel_topic="/opendr/simulation/human_model_generation/human_model",
                 output_3Dpose__topic="/opendr/simulation/human_model_generation/", device="cuda"):
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

        #if output_image_topic is not None:
        #    self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        #else:
        #    self.image_publisher = None

        #if pose_annotations_topic is not None:
        #    self.pose_publisher = rospy.Publisher(pose_annotations_topic, Detection2DArray, queue_size=10)
        #else:
        #    self.pose_publisher = None

        #rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

        self.bridge = ROSBridge()

        # Initialize the pose estimation
        self.model_generator = PIFuGeneratorLearner(device='cuda',checkpoint_dir=".")


    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_human_model_generation', anonymous=True)
        s = rospy.Service('human_model_generation', Mesh_vc, self.handle_human_model_generation)
        rospy.loginfo("Human model generation node started!")
        rospy.spin()

    def handle_human_model_generation(self, img_rgb, img_msk, extract_pose):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        rospy.wait_for_service('human_model_generation')
	
        # Convert sensor_msgs.msg.Image into OpenDR Image
        img_rgb = self.bridge.from_ros_image(img_rgb)
        img_msk = self.bridge.from_ros_image(img_msk)
        
        '''
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
        '''

if __name__ == '__main__':
    rgb_img = cv2.imread('/home/administrator/Documents/last_repo/opendr_internal/src/opendr/simulation/human_model_generation/imgs_input/rgb/result_0004.jpg')
    msk_img = cv2.imread('/home/administrator/Documents/last_repo/opendr_internal/src/opendr/simulation/human_model_generation/imgs_input/msk/result_0004.jpg')
    bridge = CvBridge()
    rgb_img_msg = bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
    msk_img_msg = bridge.cv2_to_imgmsg(msk_img, encoding="bgr8")
    rospy.wait_for_service('human_model_generation')
    try:
        human_model_gen = rospy.ServiceProxy('human_model_generation', Mesh_vc)
        extract_pose=Bool()
        extract_pose.data = True
        resp1 = human_model_gen(rgb_img_msg, msk_img_msg, extract_pose)
        #print(resp1)
    except rospy.ServiceException as e:
       print("Service call failed: %s"%e)
