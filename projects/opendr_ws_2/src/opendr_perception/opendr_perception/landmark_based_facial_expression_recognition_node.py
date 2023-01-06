#!/usr/bin/env python
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

import argparse
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge

from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner
from opendr.perception.facial_expression_recognition import landmark_extractor
from opendr.perception.facial_expression_recognition import gen_muscle_data
from opendr.perception.facial_expression_recognition import data_normalization


class LandmarkFacialExpressionRecognitionNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw",
                 output_category_topic="/opendr/landmark_expression_recognition",
                 output_category_description_topic="/opendr/landmark_expression_recognition_description",
                 device="cpu", model='pstbln_afew', shape_predictor='./predictor_path'):
        """
        Creates a ROS2 Node for landmark-based facial expression recognition.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_category_topic: Topic to which we are publishing the recognized facial expression category info
        (if None, we are not publishing the info)
        :type output_category_topic: str
        :param output_category_description_topic: Topic to which we are publishing the description of the recognized
        facial expression (if None, we are not publishing the description)
        :type output_category_description_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model:  model to use for landmark-based facial expression recognition.
         (Options: 'pstbln_ck+', 'pstbln_casia', 'pstbln_afew')
        :type model: str
        :param shape_predictor: pretrained model to use for landmark extraction from a facial image
        :type model: str
        """
        super().__init__('opendr_landmark_based_facial_expression_recognition_node')
        # Set up ROS topics and bridge

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_category_topic is not None:
            self.hypothesis_publisher = self.create_publisher(ObjectHypothesis, output_category_topic, 1)
        else:
            self.hypothesis_publisher = None

        if output_category_description_topic is not None:
            self.string_publisher = self.create_publisher(String, output_category_description_topic, 1)
        else:
            self.string_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the landmark-based facial expression recognition
        if model == 'pstbln_ck+':
            num_point = 303
            num_class = 7
        elif model == 'pstbln_casia':
            num_point = 309
            num_class = 6
        elif model == 'pstbln_afew':
            num_point = 312
            num_class = 7
        self.model_name, self.dataset_name = model.split("_")
        self.expression_classifier = ProgressiveSpatioTemporalBLNLearner(device=device, dataset_name=self.dataset_name,
                                                                         num_class=num_class, num_point=num_point,
                                                                         num_person=1, in_channels=2,
                                                                         blocksize=5, topology=[15, 10, 15, 5, 5, 10])
        model_saved_path = "./pretrained_models/" + model
        self.expression_classifier.load(model_saved_path, model)
        self.shape_predictor = shape_predictor

        self.get_logger().info("landmark-based facial expression recognition node started!")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        landmarks = landmark_extractor(image, './landmarks.npy', self.shape_predictor)

        # 3: sequence numpy data generation from extracted landmarks and normalization:

        numpy_data = _landmark2numpy(landmarks)
        norm_data = data_normalization(numpy_data)
        muscle_data = gen_muscle_data(norm_data, './muscle_data')

        # Run expression recognition
        category = self.expression_classifier.infer(muscle_data)

        if self.hypothesis_publisher is not None:
            self.hypothesis_publisher.publish(self.bridge.to_ros_category(category))

        if self.string_publisher is not None:
            self.string_publisher.publish(self.bridge.to_ros_category_description(category))


def _landmark2numpy(landmarks):
    num_landmarks = 68
    num_dim = 2  # feature dimension for each facial landmark
    num_faces = 1  # number of faces in each frame
    num_frames = 15
    numpy_data = np.zeros((1, num_dim, num_frames, num_landmarks, num_faces))
    for t in range(num_frames):
        numpy_data[0, 0:num_dim, t, :, 0] = landmarks
    return numpy_data


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_category_topic", help="Topic name for output recognized category",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/landmark_expression_recognition")
    parser.add_argument("-d", "--output_category_description_topic", help="Topic name for category description",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/landmark_expression_recognition_description")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model to use, either 'pstbln_ck+', 'pstbln_casia', 'pstbln_afew'",
                        type=str, default="pstbln_afew", choices=['pstbln_ck+', 'pstbln_casia', 'pstbln_afew'])
    parser.add_argument("-s", "--shape_predictor", help="Shape predictor (landmark_extractor) to use",
                        type=str, default='./predictor_path')
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    landmark_expression_estimation_node = \
        LandmarkFacialExpressionRecognitionNode(
            input_rgb_image_topic=args.input_rgb_image_topic,
            output_category_topic=args.output_category_topic,
            output_category_description_topic=args.output_category_description_topic,
            device=device, model=args.model,
            shape_predictor=args.shape_predictor)

    rclpy.spin(landmark_expression_estimation_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    landmark_expression_estimation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
