#!/usr/bin/env python
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

import argparse
import torch
import numpy as np
import cv2
from torchvision import transforms
import PIL

import rospy
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.perception.facial_expression_recognition import FacialEmotionLearner
from opendr.perception.facial_expression_recognition import image_processing

INPUT_IMAGE_SIZE = (96, 96)
INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]


class FacialEmotionEstimationNode:
    def __init__(self,
                 input_rgb_image_topic="/usb_cam/image_raw",
                 output_emotions_topic="/opendr/facial_emotion_estimation",
                 output_emotions_description_topic="/opendr/facial_emotion_estimation_description",
                 device="cuda"):
        """
        Creates a ROS Node for facial emotion estimation.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_emotions_topic: Topic to which we are publishing the facial emotion results
        (if None, we are not publishing the info)
        :type output_emotions_topic: str
        :param output_emotions_description_topic: Topic to which we are publishing the description of the estimated
        facial emotion (if None, we are not publishing the description)
        :type output_emotions_description_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        # Set up ROS topics and bridge
        self.input_rgb_image_topic = input_rgb_image_topic
        self.bridge = ROSBridge()

        if output_emotions_topic is not None:
            self.hypothesis_publisher = rospy.Publisher(output_emotions_topic, ObjectHypothesis, queue_size=1)
        else:
            self.hypothesis_publisher = None

        if output_emotions_description_topic is not None:
            self.string_publisher = rospy.Publisher(output_emotions_description_topic, String, queue_size=1)
        else:
            self.string_publisher = None

        # Initialize the facial emotion estimator
        self.facial_emotion_estimator = FacialEmotionLearner(device=device, batch_size=2,
                                                             ensemble_size=9,
                                                             name_experiment='esr_9')
        self.facial_emotion_estimator.init_model(num_branches=9)

        model_saved_path = self.facial_emotion_estimator.download(path=None, mode="pretrained")
        self.facial_emotion_estimator.load(ensemble_size=9, path_to_saved_network=model_saved_path)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_facial_emotion_estimation_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Facial emotion estimation node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        if image is None:
            return

        greyscale_image = image_processing.convert_bgr_to_grey(image)
        # Detect face
        face_coordinates = detect_face(greyscale_image)

        if face_coordinates is None:
            return
        else:
            face = image[face_coordinates[0][1]:face_coordinates[1][1], face_coordinates[0][0]:face_coordinates[1][0], :]

            # Pre_process detected face
            input_face = _pre_process_input_image(face)

            # Recognize facial expression
            emotion, affect = self.facial_emotion_estimator.infer(input_face)

        if self.hypothesis_publisher is not None:
            self.hypothesis_publisher.publish(self.bridge.to_ros_category(emotion))

        if self.string_publisher is not None:
            self.string_publisher.publish(self.bridge.to_ros_category_description(emotion))


def detect_face(image):
    """
    Detects faces in an image.

    :param image: (ndarray) Raw input image.
    :return: (list) Tuples with coordinates of a detected face.
    """

    # Converts to greyscale
    greyscale_image = image_processing.convert_bgr_to_grey(image)

    _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("python/perception/facial_expression_recognition/"
                                                        "image_based_facial_expression_recognition/"
                                                        "face_detector/frontal_face.xml")
    faces = _FACE_DETECTOR_HAAR_CASCADE.detectMultiScale(greyscale_image, scaleFactor=1.2, minNeighbors=9,
                                                         minSize=(60, 60))
    face_coordinates = [[[x, y], [x + w, y + h]] for (x, y, w, h) in faces] if not (faces is None) else []
    face_coordinates = np.array(face_coordinates)

    # Returns None if no face is detected
    return face_coordinates[0] if (len(face_coordinates) > 0 and (np.sum(face_coordinates[0]) > 0)) else None


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_rgb_image_topic', type=str, help='Topic name for input rgb image',
                        default='/usb_cam/image_raw')
    parser.add_argument("-o", "--output_emotions_topic", help="Topic name for output emotion",
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

    facial_emotion_estimation_node = FacialEmotionEstimationNode(
        input_rgb_image_topic=args.input_rgb_image_topic,
        output_emotions_topic=args.output_emotions_topic,
        output_emotions_description_topic=args.output_emotions_description_topic,
        device=device)

    facial_emotion_estimation_node.listen()
