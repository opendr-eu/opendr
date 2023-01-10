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

import cv2
import os
import argparse
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from cv_bridge import CvBridge

from opendr.projects.python.simulation.synthetic_multi_view_facial_image_generation.algorithm.DDFA.utils.ddfa \
    import str2bool
from opendr.src.opendr.engine.data import Image
from opendr.projects.python.simulation.synthetic_multi_view_facial_image_generation.SyntheticDataGeneration \
    import MultiviewDataGeneration


class SyntheticDataGeneratorNode(Node):

    def __init__(self, args, input_rgb_image_topic="/image_raw",
                 output_rgb_image_topic="/opendr/synthetic_facial_images"):
        """
        Creates a ROS Node for SyntheticDataGeneration
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the synthetic facial image (if None, no image
        is published)
        :type output_rgb_image_topic: str
        """
        super().__init__('synthetic_facial_image_generation_node')
        self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 10)
        self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)
        self._cv_bridge = CvBridge()
        self.ID = 0
        self.args = args
        self.path_in = args.path_in
        self.key = str(args.path_3ddfa + "/example/Images/")
        self.key1 = str(args.path_3ddfa + "/example/")
        self.key2 = str(args.path_3ddfa + "/results/")
        self.save_path = args.save_path
        self.val_yaw = args.val_yaw
        self.val_pitch = args.val_pitch
        self.device = args.device

        # Initialize the SyntheticDataGeneration
        self.synthetic = MultiviewDataGeneration(self.args)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image

        cv_image = self._cv_bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        self.ID = self.ID + 1
        # Get an OpenCV image back
        image = cv2.cvtColor(image.opencv(), cv2.COLOR_RGBA2BGR)
        name = str(f"{self.ID:02d}" + "_single.jpg")
        cv2.imwrite(os.path.join(self.path_in, name), image)

        if self.ID == 10:
            # Run SyntheticDataGeneration
            self.synthetic.eval()
            self.ID = 0
            # Annotate image and publish results
            current_directory_path = os.path.join(self.save_path, str("/Documents_orig/"))
            for file in os.listdir(current_directory_path):
                name, ext = os.path.splitext(file)
                if ext == ".jpg":
                    image_file_savepath = os.path.join(current_directory_path, file)
                    cv_image = cv2.imread(image_file_savepath)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    if self.image_publisher is not None:
                        image = Image(np.array(cv_image, dtype=np.uint8))
                        message = self.bridge.to_ros_image(image, encoding="rgb8")
                        self.image_publisher.publish(message)
            for f in os.listdir(self.path_in):
                os.remove(os.path.join(self.path_in, f))


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=str, default="/opendr/synthetic_facial_images")
    parser.add_argument("--path_in", default=os.path.join("opendr", "projects",
                                                          "data_generation",
                                                          "synthetic_multi_view_facial_image_generation",
                                                          "demos", "imgs_input"),
                        type=str, help='Give the path of image folder')
    parser.add_argument('--path_3ddfa', default=os.path.join("opendr", "projects",
                                                             "data_generation",
                                                             "synthetic_multi_view_facial_image_generation",
                                                             "algorithm", "DDFA"),
                        type=str, help='Give the path of DDFA folder')
    parser.add_argument('--save_path', default=os.path.join("opendr", "projects",
                                                            "data_generation",
                                                            "synthetic_multi_view_facial_image_generation",
                                                            "results"),
                        type=str, help='Give the path of results folder')
    parser.add_argument('--val_yaw', default="10 20", nargs='+', type=str, help='yaw poses list between [-90,90]')
    parser.add_argument('--val_pitch', default="30 40", nargs='+', type=str, help='pitch poses list between [-90,90]')
    parser.add_argument("--device", default="cuda", type=str, help="choose between cuda or cpu ")
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--dump_res', default='true', type=str2bool,
                        help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='true', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
    parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
    parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
    parser.add_argument('--save_dir', default='./algorithm/DDFA/results', type=str, help='dir to save result')
    parser.add_argument('--save_lmk_dir', default='./example', type=str, help='dir to save landmark result')
    parser.add_argument('--img_list', default='./txt_name_batch.txt', type=str, help='test image list file')
    parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
    parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
    parser.add_argument('--resume_idx', default=0, type=int)
    args = parser.parse_args()

    synthetic_data_generation_node = SyntheticDataGeneratorNode(args=args,
                                                                input_rgb_image_topic=args.input_rgb_image_topic,
                                                                output_rgb_image_topic=args.output_rgb_image_topic)

    rclpy.spin(synthetic_data_generation_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    synthetic_data_generation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
