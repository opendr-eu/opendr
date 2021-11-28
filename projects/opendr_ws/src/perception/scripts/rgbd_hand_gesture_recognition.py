#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from opendr.engine.data import Image
from vision_msgs.msg import Classification2D, ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from std_msgs.msg import Header
from opendr_bridge import ROSBridge
import os
from opendr.perception.multimodal_human_centric.rgbd_hand_gesture_learner.rgbd_hand_gesture_learner import \
    RgbdHandGestureLearner
import message_filters
import cv2
class RgbdHandGestureNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", input_depth_image_topic="/usb_cam/image_raw",
                 gesture_annotations_topic="/opendr/gestures", device="cuda"):
        """
        Creates a ROS Node for gesture recognition from RGBD
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param input_depth_image_topic: Topic from which we are reading the input depth image
        :type input_depth_image_topic: str
        :param gesture_annotations_topic: Topic to which we are publishing the predicted gesture class
        :type gesture_annotations_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        self.gesture_publisher = rospy.Publisher(gesture_annotations_topic, Classification2D, queue_size=10)


        image_sub = message_filters.Subscriber(input_image_topic, ROS_Image)
        depth_sub = message_filters.Subscriber(input_depth_image_topic, ROS_Image)
        #synchronize image and depth data topics
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
        ts.registerCallback(self.callback)
        
        self.bridge = ROSBridge()

        # Initialize the gesture recognition
        self.gesture_learner = RgbdHandGestureLearner(n_class=16, architecture="mobilenet_v2", device=device)
        model_path = './mobilenet_v2'
        if not os.path.exists(model_path):
            self.gesture_learner.download(path=model_path)
        self.gesture_learner.load(path=model_path)
        
        #mean and std for preprocessing, based on HANDS dataset
        self.mean = np.asarray([0.485, 0.456, 0.406, 0.0303]).reshape(1, 1, 4)
        self.std = np.asarray([0.229, 0.224, 0.225, 0.0353]).reshape(1, 1, 4) 
        

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_gesture_recognition', anonymous=True)
        rospy.loginfo("RGBD gesture recognition node started!")
        rospy.spin()

    def callback(self, image_data, depth_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param image_data: input image message
        :type image_data: sensor_msgs.msg.Image
        :param depth_data: input depth image message
        :type depth_data: sensor_msgs.msg.Image
        """
        
        # Convert sensor_msgs.msg.Image into OpenDR Image and preprocess
        image = self.bridge.from_ros_image(image_data, encoding='bgr8')
        depth_data.encoding='mono16'
        depth_image = from_ros_image_to_depth(depth_data, encoding='mono16')
        img = self.preprocess(image, depth_image)
        
        # Run gesture recognition
        gesture_class = self.gesture_learner.infer(img)

        #  Publish results
        ros_gesture = from_category(gesture_class)
        self.gesture_publisher.publish(ros_gesture)
        
    def preprocess(self, image, depth_img):
        '''
        Preprocess image, depth_image and concatenate them
        :param image_data: input image 
        :type image_data: engine.data.Image
        :param depth_data: input depth image 
        :type depth_data: engine.data.Image
        '''
        image = image.numpy() / (2**8 - 1)
        depth_img = depth_img.numpy() / (2**16 - 1)

        # resize the images to 224x224
        image = cv2.resize(image, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))

        # concatenate and standardize
        img = np.concatenate([image, np.expand_dims(depth_img, axis=-1)], axis = -1)
        img = (img - self.mean) / self.std
        
        img = Image(img, dtype=np.float32)
        return img


def from_category(prediction, source_data=None):
    '''
    Helper function that wraps model output to class prediction wrapped into Classification2D message
    :param prediction: output of gesture learner
    :param source_data: original image or None
    '''
    classification = Classification2D()
    classification.header = Header()
    classification.header.stamp = rospy.get_rostime()

    result = ObjectHypothesis()
    result.id = prediction.data
    result.score = prediction.confidence
    classification.results.append(result)
    if source_data is not None:
        classification.source_img = source_data
    return classification
            
def from_ros_image_to_depth(message, encoding):
    """
    Converts a ROS image message into an OpenDR depth image
    :param message: ROS image to be converted
    :type message: sensor_msgs.msg.Image
    :param encoding: encoding to be used for the conversion
    :type encoding: str
    """
    cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
    cv_image = np.expand_dims(cv_image, axis=-1)
    image = Image(np.asarray(cv_image, dtype=np.uint8))
    return image            
    
if __name__ == '__main__':
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    #default topics are according to kinectv2 drivers at https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai-iai_kinect2
    depth_topic = "/kinect2/qhd/image_depth_rect"
    image_topic = "/kinect2/qhd/image_color_rect"
    gesture_node = RgbdHandGestureNode(input_image_topic=image_topic, input_depth_image_topic=depth_topic,
                                              device=device)
    gesture_node.listen()
