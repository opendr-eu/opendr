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
import cv2
from torchvision import transforms
import PIL

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.facial_expression_recognition import FacialEmotionLearner
from opendr.perception.facial_expression_recognition import image_processing
from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray

INPUT_IMAGE_SIZE = (96, 96)
INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]


class FacialEmotionEstimationNode(Node):
    def __init__(self,
                 face_detector_learner,
                 input_rgb_image_topic="/image_raw",
                 output_rgb_image_topic="/opendr/image_emotion_estimation_annotated",
                 output_emotions_topic="/opendr/facial_emotion_estimation",
                 output_emotions_description_topic="/opendr/facial_emotion_estimation_description",
                 device="cuda"):
        """
        Creates a ROS Node for facial emotion estimation.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param output_emotions_topic: Topic to which we are publishing the facial emotion results
        (if None, we are not publishing the info)
        :type output_emotions_topic: str
        :param output_emotions_description_topic: Topic to which we are publishing the description of the estimated
        facial emotion (if None, we are not publishing the description)
        :type output_emotions_description_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__('opendr_facial_emotion_estimation_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)
        self.bridge = ROS2Bridge()

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if output_emotions_topic is not None:
            self.hypothesis_publisher = self.create_publisher(ObjectHypothesis, output_emotions_topic, 1)
        else:
            self.hypothesis_publisher = None

        if output_emotions_description_topic is not None:
            self.string_publisher = self.create_publisher(String, output_emotions_description_topic, 1)
        else:
            self.string_publisher = None

        # Initialize the face detector
        self.face_detector = face_detector_learner

        # Initialize the facial emotion estimator
        self.facial_emotion_estimator = FacialEmotionLearner(device=device, batch_size=2,
                                                             ensemble_size=9,
                                                             name_experiment='esr_9')
        self.facial_emotion_estimator.init_model(num_branches=9)
        model_saved_path = self.facial_emotion_estimator.download(path=None, mode="pretrained")
        self.facial_emotion_estimator.load(ensemble_size=9, path_to_saved_network=model_saved_path)

        self.get_logger().info("Facial emotion estimation node started.")

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8').opencv()
        emotion = None
        # Run face detection and emotion estimation

        if image is not None:
            bounding_boxes = self.face_detector.infer(image)
            if bounding_boxes:
                bounding_boxes = BoundingBoxListToNumpyArray()(bounding_boxes)
                boxes = bounding_boxes[:, :4]
                for idx, box in enumerate(boxes):
                    (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    face_crop = image[startY:endY, startX:endX]

                    # Preprocess detected face
                    input_face = _pre_process_input_image(face_crop)

                    # Recognize facial expression
                    emotion, affect = self.facial_emotion_estimator.infer(input_face)

                    # Converts from Tensor to ndarray
                    affect = np.array([a.cpu().detach().numpy() for a in affect])
                    affect = affect[0]  # a numpy array of valence and arousal values
                    emotion = emotion[0]  # the emotion class with confidence tensor

                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), thickness=2)
                    cv2.putText(image, "Valence: %.2f" % affect[0], (startX, endY - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, "Arousal: %.2f" % affect[1], (startX, endY - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, emotion.description, (startX, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if self.hypothesis_publisher is not None and emotion:
            self.hypothesis_publisher.publish(self.bridge.to_ros_category(emotion))

        if self.string_publisher is not None and emotion:
            self.string_publisher.publish(self.bridge.to_ros_category_description(emotion))

        if self.image_publisher is not None:
            # Convert the annotated OpenDR image to ROS image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def _pre_process_input_image(image):
    """
    Pre-processes an image for ESR-9.

    :param image: (ndarray)
    :return: (ndarray) image
    """

    image = image_processing.resize(image, INPUT_IMAGE_SIZE)
    image = PIL.Image.fromarray(image)
    image = transforms.Normalize(mean=INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)

    return image


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_rgb_image_topic', type=str, help='Topic name for input rgb image',
                        default='/image_raw')
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_emotion_estimation_annotated")
    parser.add_argument("-e", "--output_emotions_topic", help="Topic name for output emotion",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/facial_emotion_estimation")
    parser.add_argument('-m', '--output_emotions_description_topic',
                        help='Topic to which we are publishing the description of the estimated facial emotion',
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/facial_emotion_estimation_description")
    parser.add_argument('-d', '--device', help='Device to use, either cpu or cuda',
                        type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            print("GPU found.")
            device = 'cuda'
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU")
            device = 'cpu'
    except:
        print("Using CPU")
        device = 'cpu'

    # Initialize the face detector
    face_detector = RetinaFaceLearner(backbone="resnet", device=device)
    face_detector.download(path=".", verbose=True)
    face_detector.load("retinaface_{}".format("resnet"))

    facial_emotion_estimation_node = FacialEmotionEstimationNode(
        face_detector,
        input_rgb_image_topic=args.input_rgb_image_topic,
        output_rgb_image_topic=args.output_rgb_image_topic,
        output_emotions_topic=args.output_emotions_topic,
        output_emotions_description_topic=args.output_emotions_description_topic,
        device=device)

    rclpy.spin(facial_emotion_estimation_node)
    facial_emotion_estimation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
