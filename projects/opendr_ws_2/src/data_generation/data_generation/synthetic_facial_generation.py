#!/usr/bin/env python3.6
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

import cv2
import torch
import rclpy
from rclpy.node import Node
from ros2_bridge.bridge import ROS2Bridge
import numpy as np
from sensor_msgs.msg import Image as ROS_Image
from SyntheticDataGeneration import MultiviewDataGeneration
import os
import argparse
from opendr.engine.data import Image
from algorithm.DDFA.utils.ddfa import str2bool


class SyntheticDataGeneration(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/synthetic_facial_images",
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
            self.image_publisher = self.create_publisher(output_image_topic, ROS_Image, queue_size=10)
        else:
            self.image_publisher = None
        self.create_subscription(input_image_topic, ROS_Image, self.callback)
        self.bridge = ROS2Bridge()
        self.ID = 0

        # Initialize the SyntheticDataGeneration
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-device", default="cuda", type=str, help="choose between cuda or cpu ")
        self.parser.add_argument("-path_in", default=os.path.join("opendr_internal", "projects",
                                                                  "data_generation",
                                                                  "",
                                                                  "demos", "imgs_input"),
                                 type=str, help='Give the path of image folder')
        self.parser.add_argument('-path_3ddfa', default=os.path.join("opendr_internal", "projects",
                                                                     "data_generation",
                                                                     "",
                                                                     "algorithm", "DDFA"),
                                 type=str, help='Give the path of DDFA folder')
        self.parser.add_argument('-save_path', default=os.path.join("opendr_internal", "projects",
                                                                    "data_generation",
                                                                    "",
                                                                    "results"),
                                 type=str, help='Give the path of results folder')
        self.parser.add_argument('-val_yaw', default="10 20", nargs='+', type=str, help='yaw poses list between [-90,90]')
        self.parser.add_argument('-val_pitch', default="30 40", nargs='+', type=str,
                                 help='pitch poses list between [-90,90] ')
        self.parser.add_argument('-f', '--files', nargs='+',
                                 help='image files paths fed into network, single or multiple images')
        self.parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
        self.parser.add_argument('--dump_res', default='true', type=str2bool,
                                 help='whether write out the visualization image')
        self.parser.add_argument('--dump_vertex', default='false', type=str2bool,
                                 help='whether write out the dense face vertices to mat')
        self.parser.add_argument('--dump_ply', default='true', type=str2bool)
        self.parser.add_argument('--dump_pts', default='true', type=str2bool)
        self.parser.add_argument('--dump_roi_box', default='false', type=str2bool)
        self.parser.add_argument('--dump_pose', default='true', type=str2bool)
        self.parser.add_argument('--dump_depth', default='true', type=str2bool)
        self.parser.add_argument('--dump_pncc', default='true', type=str2bool)
        self.parser.add_argument('--dump_paf', default='true', type=str2bool)
        self.parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
        self.parser.add_argument('--dump_obj', default='true', type=str2bool)
        self.parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
        self.parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                                 help='whether use dlib landmark to crop image')
        self.parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
        self.parser.add_argument('--bbox_init', default='two', type=str,
                                 help='one|two: one-step bbox initialization or two-step')
        self.parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
        self.parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
        self.parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
        self.parser.add_argument('--save_dir', default='./algorithm/DDFA/results', type=str, help='dir to save result')
        self.parser.add_argument('--save_lmk_dir', default='./example', type=str, help='dir to save landmark result')
        self.parser.add_argument('--img_list', default='./txt_name_batch.txt', type=str, help='test image list file')
        self.parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
        self.parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
        self.parser.add_argument('--resume_idx', default=0, type=int)
        self.args = self.parser.parse_args()
        self.synthetic = MultiviewDataGeneration(self.args)

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


def main(args=None):
    rclpy.init(args=args)
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

    rclpy.spin(syntheticdatageneration_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    syntheticdatageneration_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
