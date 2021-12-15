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

import cv2
import torch
import os
from opendr.engine.target import TrackingAnnotation
import rospy
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
    ObjectTracking2DFairMotLearner,
)
from opendr.engine.data import Image


class ObjectTracking2DFairMotNode:
    def __init__(
        self,
        input_image_topic="/usb_cam/image_raw",
        output_detection_topic="/opendr/detection",
        output_tracking_id_topic="/opendr/tracking_id",
        output_image_topic="/opendr/image_annotated",
        device="cuda:0",
        model_name="fairmot_dla34",
        temp_dir="temp",
    ):
        """
        Creates a ROS Node for 2D object tracking
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param output_detection_topic: Topic to which we are publishing the detections
        :type output_detection_topic:  str
        :param output_tracking_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model_name: the pretrained model to download or a saved model in temp_dir folder to use
        :type model_name: str
        :param temp_dir: the folder to download models
        :type temp_dir: str
        """

        # # Initialize the face detector
        self.learner = ObjectTracking2DFairMotLearner(
            device=device, temp_path=temp_dir,
        )
        if not os.path.exists(os.path.join(temp_dir, model_name)):
            ObjectTracking2DFairMotLearner.download(model_name, temp_dir)

        self.learner.load(os.path.join(temp_dir, model_name), verbose=True)

        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        self.detection_publisher = rospy.Publisher(
            output_detection_topic, Detection2DArray, queue_size=10
        )
        self.tracking_id_publisher = rospy.Publisher(
            output_tracking_id_topic, Int32MultiArray, queue_size=10
        )

        if output_image_topic is not None:
            self.output_image_publisher = rospy.Publisher(
                output_image_topic, ROS_Image, queue_size=10
            )

        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding="bgr8")
        tracking_boxes = self.learner.infer(image)

        if self.output_image_publisher is not None:
            frame = image.opencv()
            draw_predictions(frame, tracking_boxes)
            message = self.bridge.to_ros_image(
                Image(frame), encoding="bgr8"
            )
            self.output_image_publisher.publish(message)
            rospy.loginfo("Published annotated image")

        detection_boxes = tracking_boxes.bounding_box_list()
        ids = [tracking_box.id for tracking_box in tracking_boxes]

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes(detection_boxes)
        if self.detection_publisher is not None:
            self.detection_publisher.publish(ros_boxes)
            rospy.loginfo("Published detection boxes")

        ros_ids = Int32MultiArray()
        ros_ids.data = ids

        if self.tracking_id_publisher is not None:
            self.tracking_id_publisher.publish(ros_ids)
            rospy.loginfo("Published tracking ids")


colors = [
    (255, 0, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (35, 69, 55),
    (43, 63, 54),
]


def draw_predictions(frame, predictions: TrackingAnnotation, is_centered=False, is_flipped_xy=True):
    global colors
    w, h, _ = frame.shape

    for prediction in predictions.boxes:
        prediction = prediction

        if not hasattr(prediction, "id"):
            prediction.id = 0

        color = colors[int(prediction.id) * 7 % len(colors)]

        x = prediction.left
        y = prediction.top

        if is_flipped_xy:
            x = prediction.top
            y = prediction.left

        if is_centered:
            x -= prediction.width
            y -= prediction.height

        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (
                int(x + prediction.width),
                int(y + prediction.height),
            ),
            color,
            2,
        )


if __name__ == "__main__":
    # Automatically run on GPU/CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # initialize ROS node
    rospy.init_node("opendr_fair_mot", anonymous=True)
    rospy.loginfo("FairMOT node started")

    model_name = rospy.get_param("~model_name", "fairmot_dla34")
    temp_dir = rospy.get_param("~temp_dir", "temp")
    input_image_topic = rospy.get_param(
        "~input_image_topic", "/opendr/dataset_image"
    )
    rospy.loginfo("Using model_name: {}".format(model_name))

    # created node object
    fair_mot_node = ObjectTracking2DFairMotNode(
        device=device,
        model_name=model_name,
        input_image_topic=input_image_topic,
        temp_dir=temp_dir,
    )
    # begin ROS communications
    rospy.spin()
