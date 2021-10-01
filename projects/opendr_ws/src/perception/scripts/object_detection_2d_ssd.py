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
import mxnet as mx
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes


class ObjectDetectionSSDNode:
    def __init__(self, input_image_topic="/usb_cam/image_raw", output_image_topic="/opendr/image_boxes_annotated",
                 detections_topic="/opendr/objects", device="cuda", backbone="vgg16_atrous"):
        """
        Creates a ROS Node for face detection
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated pose annotations)
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: backbone network
        :type backbone: str
        """

        # Initialize the face detector
        self.object_detector = SingleShotDetectorLearner(backbone=backbone, device=device)
        self.object_detector.download(path=".", verbose=True)
        self.object_detector.load("ssd_default_person")
        self.class_names = self.object_detector.classes

        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        # setup communications
        if output_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.bbox_publisher = rospy.Publisher(detections_topic, Detection2DArray, queue_size=10)
        else:
            self.bbox_publisher = None

        rospy.Subscriber(input_image_topic, ROS_Image, self.callback)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data)

        # Run pose estimation
        boxes = self.object_detector.infer(image, threshold=0.45, keep_size=False)

        # Get an OpenCV image back
        image = np.float32(image.numpy())

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes(boxes)
        if self.bbox_publisher is not None:
            self.bbox_publisher.publish(ros_boxes)
            rospy.loginfo("Published face boxes")

        # Annotate image and publish result
        # NOTE: converting back to OpenDR BoundingBoxList is unnecessary here,
        # only used to test the corresponding bridge methods
        odr_boxes = self.bridge.from_ros_boxes(ros_boxes)
        image = draw_bounding_boxes(image, odr_boxes, class_names=self.class_names)
        if self.image_publisher is not None:
            message = self.bridge.to_ros_image(np.uint8(image))
            self.image_publisher.publish(message)
            rospy.loginfo("Published annotated image")


if __name__ == '__main__':
    # Automatically run on GPU/CPU
    try:
        if mx.context.num_gpus() > 0:
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'

    # initialize ROS node
    rospy.init_node('opendr_object_detection', anonymous=True)
    rospy.loginfo("Object detection node started!")

    input_image_topic = rospy.get_param("~input_image_topic", "/videofile/image_raw")

    # created node object
    object_detection_node = ObjectDetectionSSDNode(device=device, input_image_topic=input_image_topic)
    # begin ROS communications
    rospy.spin()
