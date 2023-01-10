#!/usr/bin/env python3
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
import cv2
import torch

import rospy
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.perception.face_recognition import FaceRecognitionLearner
from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray


class FaceRecognitionNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw",
                 output_rgb_image_topic="/opendr/image_face_reco_annotated",
                 detections_topic="/opendr/face_recognition", detections_id_topic="/opendr/face_recognition_id",
                 database_path="./database", device="cuda", backbone="mobilefacenet"):
        """
        Creates a ROS Node for face recognition.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the recognized face information (if None,
        no face recognition message is published)
        :type detections_topic:  str
        :param detections_id_topic: Topic to which we are publishing the ID of the recognized person (if None,
        no ID message is published)
        :type detections_id_topic:  str
        :param device: Device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: Backbone network
        :type backbone: str
        :param database_path: Path of the directory where the images of the faces to be recognized are stored
        :type database_path: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_rgb_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.face_publisher = rospy.Publisher(detections_topic, ObjectHypothesis, queue_size=1)
        else:
            self.face_publisher = None

        if detections_id_topic is not None:
            self.face_id_publisher = rospy.Publisher(detections_id_topic, String, queue_size=1)
        else:
            self.face_id_publisher = None

        self.bridge = ROSBridge()

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

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_face_recognition_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Face recognition node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        # Get an OpenCV image back
        image = image.opencv()

        # Run face detection and recognition
        if image is not None:
            bounding_boxes = self.face_detector.infer(image)
            if bounding_boxes:
                bounding_boxes = BoundingBoxListToNumpyArray()(bounding_boxes)
                boxes = bounding_boxes[:, :4]
                for idx, box in enumerate(boxes):
                    (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    frame = image[startY:endY, startX:endX]
                    result = self.recognizer.infer(frame)

                    # Publish face information and ID
                    if self.face_publisher is not None:
                        self.face_publisher.publish(self.bridge.to_ros_face(result))

                    if self.face_id_publisher is not None:
                        self.face_id_publisher.publish(self.bridge.to_ros_face_id(result))

                    if self.image_publisher is not None:
                        if result.description != 'Not found':
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                        # Annotate image with face detection/recognition boxes
                        cv2.rectangle(image, (startX, startY), (endX, endY), color, thickness=2)
                        cv2.putText(image, result.description, (startX, endY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, color, 2, cv2.LINE_AA)

            if self.image_publisher is not None:
                # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
                self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_face_reco_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/face_recognition")
    parser.add_argument("-id", "--detections_id_topic", help="Topic name for detection ID messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/face_recognition_id")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", help="Backbone network, defaults to mobilefacenet",
                        type=str, default="mobilefacenet", choices=["mobilefacenet"])
    parser.add_argument("--dataset_path",
                        help="Path of the directory where the images of the faces to be recognized are stored, "
                             "defaults to \"./database\"",
                        type=str, default="./database")
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

    face_recognition_node = FaceRecognitionNode(device=device, backbone=args.backbone, database_path=args.dataset_path,
                                                input_rgb_image_topic=args.input_rgb_image_topic,
                                                output_rgb_image_topic=args.output_rgb_image_topic,
                                                detections_topic=args.detections_topic,
                                                detections_id_topic=args.detections_id_topic)
    face_recognition_node.listen()


if __name__ == '__main__':
    main()
