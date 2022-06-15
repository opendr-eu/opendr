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


import rospy
import torch
import numpy as np
import cv2
from torchvision import transforms
import PIL
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.facial_expression_recognition import FacialEmotionLearner
from opendr.perception.facial_expression_recognition import image_processing
from opendr.perception.facial_expression_recognition import ESR
from opendr.engine.constants import OPENDR_SERVER_URL

# Haar cascade parameters
_HAAR_SCALE_FACTOR = 1.2
_HAAR_NEIGHBORS = 9
_HAAR_MIN_SIZE = (60, 60)

# Face detector method
_FACE_DETECTOR_HAAR_CASCADE = None

class FacialEmotionEstimationNode:
    def __init__(self,
                 input_image_topic="/usb_cam/image_raw",
                 output_emotions_topic="/opendr/facial_emotion",
                 output_emotions_description_topic="/opendr/facial_emotion_estimation_description",
                 device="cuda"):
        """
        Creates a ROS Node for facial emotion estimation
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
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

        if output_emotions_topic is not None:
            self.hypothesis_publisher = rospy.Publisher(output_emotions_topic, ObjectHypothesis, queue_size=10)
        else:
            self.hypothesis_publisher = None

        if output_emotions_description_topic is not None:
            self.string_publisher = rospy.Publisher(output_emotions_description_topic, String, queue_size=10)
        else:
            self.string_publisher = None

        self.input_image_topic = input_image_topic
        self.bridge = ROSBridge()

        # Initialize the facial emotion estimator
        self.facial_emotion_estimator = FacialEmotionLearner(device=device, batch_size=2,
                                                             ensemble_size=9,
                                                             name_experiment='esr_9')

        model_saved_path = self.facial_emotion_estimator.download(self, path=None, mode="pretrained",
                                                                  url=
                                                                  OPENDR_SERVER_URL + "perception/ensemble_based_cnn")

        self.facial_emotion_estimator.load(self, ensemble_size=9, path_to_saved_network=model_saved_path,
                                           file_name_base_network="Net-Base-Shared_Representations.pt",
                                           file_name_conv_branch="Net-Branch_{}.pt", fix_backbone=True)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_facial_emotion_estimation', anonymous=True)
        rospy.Subscriber(self.input_image_topic, ROS_Image, self.callback)
        rospy.loginfo("Facial emotion estimation node started!")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        if image is None:
            return
        # preprocess the image
        image = image_processing.read(image)

        # Detect face
        face_coordinates = detect_face(image)

        if face_coordinates is None:
            return
        else:
            face = image[face_coordinates[0][1]:face_coordinates[1][1],
                         face_coordinates[0][0]:face_coordinates[1][0], :]

            # Pre_process detected face
            input_face = _pre_process_input_image(face)
            input_face = input_face.to(device)

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

    face_coordinates = _haar_cascade_face_detection(greyscale_image, _HAAR_SCALE_FACTOR,
                                                    _HAAR_NEIGHBORS, _HAAR_MIN_SIZE)
    # Returns None if no face is detected
    return face_coordinates[0] if (len(face_coordinates) > 0 and (np.sum(face_coordinates[0]) > 0)) else None


def _haar_cascade_face_detection(image, scale_factor, neighbors, min_size):
    """
    Face detection using the Haar Feature-based Cascade Classifiers (Viola and Jones, 2004).

    References:
    Viola, P. and Jones, M. J. (2004). Robust real-time face detection.
    International journal of computer vision, 57(2), 137-154.

    :param image: (ndarray) Raw image.
    :param scale_factor: Scale factor to resize input image.
    :param neighbors: Minimum number of bounding boxes to be classified as a face.
    :param min_size: Minimum size of the face bounding box.
    :return: (ndarray) Coordinates of the detected face.
    """
    global _FACE_DETECTOR_HAAR_CASCADE

    # Verifies if haar cascade classifiers are initialized
    if _FACE_DETECTOR_HAAR_CASCADE is None:
        _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("perception/facial_expression_recognition/"
                                                            "image_based_facial_expression_recognition/"
                                                            "face_detector/haar_cascade/frontal_face.xml")

    # Runs haar cascade classifiers
    faces = _FACE_DETECTOR_HAAR_CASCADE.detectMultiScale(image, scale_factor, neighbors, minSize=min_size)

    # Gets coordinates
    face_coordinates = [[[x, y], [x + w, y + h]] for (x, y, w, h) in faces] if not (faces is None) else []

    return np.array(face_coordinates)


def _pre_process_input_image(image):
    """
    Pre-processes an image for ESR-9.

    :param image: (ndarray)
    :return: (ndarray) image
    """

    image = image_processing.resize(image, ESR.INPUT_IMAGE_SIZE)
    image = PIL.Image.fromarray(image)
    image = transforms.Normalize(mean=ESR.INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=ESR.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)

    return image


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

    facial_emotion_estimation_node = FacialEmotionEstimationNode(device=device)
    facial_emotion_estimation_node.listen()
