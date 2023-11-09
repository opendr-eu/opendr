#!/usr/bin/env python3.6
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


import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from SyntheticDataGeneration import MultiviewDataGenerationLearner
import os
import cv2
import argparse
from src.opendr.engine.data import Image


class Synthetic_Data_Generation:

    def __init__(self, input_image_topic="/usb_cam/image_raw", output_image_topic="/opendr/synthetic_facial_images",
                 device="cuda"):
        """
        Creates a ROS Node for SyntheticDataGeneration
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the synthetic facial image (if None, we are not publishing
        any image)
        :type output_image_topic: str
        :param device: device on which we are running eval ('cpu' or 'cuda')
        :type device: str
        """

        if output_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        else:
            self.image_publisher = None
        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

        self.bridge = ROSBridge()
        self.ID = 0

        # Initialize the SyntheticDataGeneration
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-path_in', default='/home/ekakalet/Pictures/TEST', type=str,
                                 help='Give the path of image folder')
        self.parser.add_argument('-path_3ddfa', default='./', type=str, help='Give the path of DDFA folder')
        self.parser.add_argument('-save_path', default='./results/', type=str, help='Give the path of results folder')
        self.parser.add_argument('-val_yaw',  default="15,-15", nargs='+', type=str,
                                 help='yaw poses list between [-90,90] ')
        self.parser.add_argument('-val_pitch', default="15,-15", nargs='+', type=str,
                                 help='pitch poses list between [-90,90] ')
        self.args = self.parser.parse_args()
        self.synthetic = MultiviewDataGenerationLearner(path_in=self.args.path_in, path_3ddfa=self.args.path_3ddfa,
                                                        save_path=self.args.save_path,
                                                        val_yaw=self.args.val_yaw, val_pitch=self.args.val_pitch)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_SyntheticDataGeneration', anonymous=True)
        rospy.loginfo("SyntheticDataGeneration node started!")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data)
        self.ID = self.ID + 1
        # Get an OpenCV image back
        image = np.float32(image.numpy())
        name = str(f"{self.ID:02d}"+"_single.jpg")
        cv2.imwrite(os.path.join(self.args.path_in, name), image)

        if (self.ID == 5):
            # Run SyntheticDataGeneration
            self.synthetic.eval()
            self.ID = 0
            # Annotate image and publish results
            current_directory_path = os.path.join(self.args.save_path, str("/Documents_orig/"))
            for file in os.listdir(current_directory_path):
                name, ext = os.path.splitext(file)
                if ext == ".jpg":
                    image_file_savepath = os.path.join(current_directory_path, file)
                    cv_image = cv2.imread(image_file_savepath)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    if self.image_publisher is not None:
                        image = Image(np.array(cv_image, dtype=np.uint8))
                        message = self.bridge.to_ros_image(image, encoding="bgr8")
                        self.image_publisher.publish(message)
        for f in os.listdir(self.args.path_in):
            os.remove(os.path.join(self.args.path_in, f))

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

    syntheticdatageneration_node = Synthetic_Data_Generation(device=device)
    syntheticdatageneration_node.listen()
