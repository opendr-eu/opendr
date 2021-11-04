#!/usr/bin/env python3.6
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
import sys

sys.path.insert(1, '/home/ekakalet/PycharmProjects/opendr_internal/projects/data_generation/synthetic-multi-view-facal-image-generation/3ddfa')
from SyntheticDataGeneration import MultiviewDataGenerationLearner
import os
import argparse
from cv_bridge import CvBridge, CvBridgeError

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

        # Initialize the SyntheticDataGeneration
        

        parser = argparse.ArgumentParser()
        parser.add_argument('-path_in', default='/home/ekakalet/Pictures/TEST', type=str, help='Give the path of image folder')
        parser.add_argument('-path_3ddfa', default='/home/ekakalet/PycharmProjects/opendr_internal/projects/data_generation/synthetic-multi-view-facial-image-generation/3ddfa', type=str, help='Give the path of 3ddfa folder')
        parser.add_argument('-save_path', default='/home/ekakalet/PycharmProjects/opendr_internal/projects/data_generation/synthetic-multi-view-facial-image-generation/results/', type=str, help='Give the path of results folder')
        parser.add_argument('-val_yaw',  default="15,-15", nargs='+', type=str, help='yaw poses list between [-90,90] ')
        parser.add_argument('-val_pitch', default="15,-15", nargs='+', type=str,  help='pitch poses list between [-90,90] ')
        args = parser.parse_args()
        synthetic = MultiviewDataGenerationLearner(path_in=args.path_in, path_3ddfa=args.path_3ddfa, save_path=args.save_path,
                                           val_yaw=args.val_yaw, val_pitch=args.val_pitch)
        self.ID=0

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
        global ID
        try:
        # Convert sensor_msgs.msg.Image into OpenDR Image
           image = self.bridge.from_ros_image(data)
           rospy.sleep(1)
        except CvBridgeError as e:
             print(e)
        else:
             exists = os.path.isfile('./camera_image'+str(self.ID)+'.png')
        if exists:
             self.ID=self.ID+1
        # Get an OpenCV image back
             image = np.float32(image.numpy())
             name= 'camera_image'+ str(self.ID)+'_single.jpg'
             cv2.imwrite(os.path.join(self.path_in, name), image)
             print("image enregistrer num ", str(self.ID))
        
        # Run SyntheticDataGeneration
        synthetic.eval()

        
        #  Annotate image and publish results
        for subdir, dirs, files in os.walk(self.save_path):
            current_directory_path = os.path.abspath(subdir)
            if (files):  
              for file in files:
                image_file_savepath= os.path.join(current_directory_path, file)  
                image = cv2.imread(image_file_savepath)  
                
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

    syntheticdatageneration_node = Synthetic_Data_Generation(device=device)
    syntheticdatageneration_node.listen()
