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
from vision_msgs.msg import ObjectHypothesis
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.perception.face_recognition import FaceRecognitionLearner
from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray


class FaceRecognitionNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw",
                 face_recognition_topic="/opendr/face_recognition",
                 face_id_topic="/opendr/face_recognition_id",
                 database_path="./database", device="cuda",
                 backbone='mobilefacenet'):
        """
        Creates a ROS Node for face recognition
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param face_recognition_topic: Topic to which we are publishing the recognized face info
        (if None, we are not publishing the info)
        :type face_recognition_topic: str
        :param face_id_topic: Topic to which we are publishing the ID of the recognized person
         (if None, we are not publishing the ID)
        :type face_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        # Initialize the face recognizer
        self.recognizer = FaceRecognitionLearner(device=device, mode='backbone_only', backbone=backbone)
        self.recognizer.download(path=".")
        self.recognizer.load(".")
        self.recognizer.fit_reference(database_path, save_path=".", create_new=True)

        # Initialize the face detector
        self.face_detector = RetinaFaceLearner(backbone='mnet', device=device)
        self.face_detector.download(path=".", verbose=True)
        self.face_detector.load("retinaface_{}".format('mnet'))
        self.class_names = ["face", "masked_face"]

        if face_recognition_topic is not None:
            self.face_publisher = rospy.Publisher(face_recognition_topic, ObjectHypothesis, queue_size=10)
        else:
            self.face_publisher = None

        if face_id_topic is not None:
            self.face_id_publisher = rospy.Publisher(face_id_topic, String, queue_size=10)
        else:
            self.face_id_publisher = None

        self.bridge = ROSBridge()
        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data)
        image = image.opencv()

        # Run face detection and recognition
        if image is not None:
            bounding_boxes = self.face_detector.infer(image)
            if bounding_boxes:
                bounding_boxes = BoundingBoxListToNumpyArray()(bounding_boxes)
                boxes = bounding_boxes[:, :4]
                for idx, box in enumerate(boxes):
                    (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    img = image[startY:endY, startX:endX]
                    result = self.recognizer.infer(img)

                    if result.data is not None:
                        if self.face_publisher is not None:
                            ros_face = self.bridge.to_ros_face(result)
                            self.face_publisher.publish(ros_face)

                        if self.face_id_publisher is not None:
                            ros_face_id = self.bridge.to_ros_face_id(result)
                            self.face_id_publisher.publish(ros_face_id.data)

                    else:
                        result.description = "Unknown"
                        if self.face_publisher is not None:
                            ros_face = self.bridge.to_ros_face(result)
                            self.face_publisher.publish(ros_face)

                        if self.face_id_publisher is not None:
                            ros_face_id = self.bridge.to_ros_face_id(result)
                            self.face_id_publisher.publish(ros_face_id.data)

                # We get can the data back using self.bridge.from_ros_face(ros_face)
                # e.g.
                # face = self.bridge.from_ros_face(ros_face)
                # face.description = self.recognizer.database[face.id][0]


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

    # initialize ROS node
    rospy.init_node('opendr_face_recognition', anonymous=True)
    rospy.loginfo("Face recognition node started!")

    # get network backbone
    backbone = rospy.get_param("~backbone", "mobilefacenet")
    input_image_topic = rospy.get_param("~input_image_topic", "/usb_cam/image_raw")
    database_path = rospy.get_param('~database_path', './')
    rospy.loginfo("Using backbone: {}".format(backbone))
    assert backbone in ["mobilefacenet", "ir_50"], "backbone should be one of ['mobilefacenet', 'ir_50']"

    face_recognition_node = FaceRecognitionNode(device=device, backbone=backbone,
                                                input_image_topic=input_image_topic,
                                                database_path=database_path)
    # begin ROS communications
    rospy.spin()
