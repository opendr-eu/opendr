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
import cv2
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.pose_estimation import get_bbox
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList


class FallDetectionNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", output_image_topic="/opendr/image_fall_annotated",
                 fall_annotations_topic="/opendr/falls", device="cuda"):
        """
        Creates a ROS Node for fall detection
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param fall_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated fall annotations)
        :type fall_annotations_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        if output_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        else:
            self.image_publisher = None

        if fall_annotations_topic is not None:
            self.fall_publisher = rospy.Publisher(fall_annotations_topic, Detection2DArray, queue_size=10)
        else:
            self.fall_publisher = None

        self.input_image_topic = input_image_topic

        self.bridge = ROSBridge()

        # Initialize the pose estimation
        self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=2,
                                                         mobilenet_use_stride=False,
                                                         half_precision=False)
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        self.fall_detector = FallDetectorLearner(self.pose_estimator)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_fall_detection', anonymous=True)
        rospy.Subscriber(self.input_image_topic, ROS_Image, self.callback)
        rospy.loginfo("Fall detection node started!")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run fall detection
        detections = self.fall_detector.infer(image)

        # Get an OpenCV image back
        image = image.opencv()

        bboxes = BoundingBoxList([])
        for detection in detections:
            fallen = detection[0].data
            pose = detection[2]

            if fallen == 1:
                color = (0, 0, 255)
                x, y, w, h = get_bbox(pose)
                bbox = BoundingBox(left=x, top=y, width=w, height=h, name=0)
                bboxes.data.append(bbox)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, "Detected fallen person", (5, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, 1, cv2.LINE_AA)

                # Convert detected boxes to ROS type and publish
                ros_boxes = self.bridge.to_ros_boxes(bboxes)
                if self.fall_publisher is not None:
                    self.fall_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            message = self.bridge.to_ros_image(Image(image), encoding='bgr8')
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

    fall_detection_node = FallDetectionNode(device=device)
    fall_detection_node.listen()
