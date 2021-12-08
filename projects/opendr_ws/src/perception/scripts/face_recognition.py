#!/usr/bin/env python
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
from vision_msgs.msg import ObjectHypothesis
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.face_recognition import FaceRecognitionLearner


class FaceRecognitionNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", face_recognition_topic="/opendr/face_recognition",
                 face_id_topic="/opendr/face_recognition_id", database_path="./database", device="cuda",
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

        if face_recognition_topic is not None:
            self.face_publisher = rospy.Publisher(face_recognition_topic, ObjectHypothesis, queue_size=10)
        else:
            self.face_publisher = None

        if face_id_topic is not None:
            self.face_id_publisher = rospy.Publisher(face_id_topic, String, queue_size=10)
        else:
            self.face_id_publisher = None

        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

        self.bridge = ROSBridge()

        # Initialize the pose estimation
        self.recognizer = FaceRecognitionLearner(device=device, mode='backbone_only', backbone=backbone)
        self.recognizer.download(path=".")
        self.recognizer.load(".")
        self.recognizer.fit_reference(database_path, save_path=".")

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_face_recognition', anonymous=True)
        rospy.loginfo("Face recognition node started!")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data)

        # Run face recognition
        if image is not None:
            result = self.recognizer.infer(image)
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

    face_recognition_node = FaceRecognitionNode(device=device)
    face_recognition_node.listen()
