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
import torch
from time import perf_counter
import rospy
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from std_msgs.msg import Float32
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.perception.gesture_recognition.gesture_recognition_learner import GestureRecognitionLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class GestureRecognitionNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw",
                 output_rgb_image_topic="/opendr/images", detections_topic="/opendr/gestures",
                 performance_topic=None, device="cuda", model="plus_m_1.5x_416", threshold=0.5):
        """
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no object detection message
        is published)
        :type detections_topic:  str
        :param performance_topic: Topic to which performance information is published
        :type performace_topis: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model: the name of the model of which we want to load the config file
        :type model: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_rgb_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.object_publisher = rospy.Publisher(detections_topic, Detection2DArray, queue_size=1)
        else:
            self.object_publisher = None

        if performance_topic is not None:
            self.performance_publisher = rospy.Publisher(performance_topic, Float32, queue_size=1)
        else:
            self.performance_publisher = None

        self.bridge = ROSBridge()

        # Initialize the gesture model
        self.gesture_model = GestureRecognitionLearner(model_to_use=model, device=device)
        self.gesture_model.download(path=".", verbose=True)
        self.gesture_model.load("./nanodet_{}".format(model))

        self.threshold = threshold

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_gesture_recognition_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Gesture recognition node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        if self.performance_publisher:
            start_time = perf_counter()
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run object detection
        boxes = self.gesture_model.infer(image, conf_threshold=self.threshold, nms_max_num=2)
        # Get an OpenCV image back
        image = image.opencv()

        # Publish detections in ROS message
        ros_boxes = self.bridge.to_ros_bounding_box_list(boxes)  # Convert to ROS boxes

        if self.performance_publisher is not None:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        if self.object_publisher is not None:
            self.object_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            # Annotate image with object detection boxes
            image = draw_bounding_boxes(image, boxes, class_names=self.gesture_model.classes)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/rgb_gesture_images_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/rgb_gestures")
    parser.add_argument("--performance_topic", help="Topic name for performance messages", type=str, default=None)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--threshold", help="Confidence threshold for prediction", type=float, default=0.5)
    parser.add_argument("--model", help="Model that config file will be used", type=str, default="plus_m_1.5x_416")
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

    gesture_recognition_node = GestureRecognitionNode(device=device, model=args.model,
                                                      input_rgb_image_topic=args.input_rgb_image_topic,
                                                      output_rgb_image_topic=args.output_rgb_image_topic,
                                                      detections_topic=args.detections_topic,
                                                      performance_topic=args.performance_topic, threshold=args.threshold)
    gesture_recognition_node.listen()


if __name__ == '__main__':
    main()
