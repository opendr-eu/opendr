#!/usr/bin/env python3
# Copyright 2020-2024 OpenDR European Project
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
import mxnet as mx
from time import perf_counter

import rospy
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.perception.object_detection_2d import Seq2SeqNMSLearner, SoftNMS, FastNMS, ClusterNMS


class ObjectDetectionSSDNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw",
                 output_rgb_image_topic="/opendr/image_objects_annotated", detections_topic="/opendr/objects",
                 performance_topic=None, device="cuda", backbone="vgg16_atrous", nms_type='default'):
        """
        Creates a ROS Node for object detection with SSD.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no object detection message
        is published)
        :type detections_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: backbone network
        :type backbone: str
        :param nms_type: type of NMS method
        :type nms_type: str
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

        # Initialize the object detector
        self.object_detector = SingleShotDetectorLearner(backbone=backbone, device=device)
        self.object_detector.download(path=".", verbose=True)
        self.object_detector.load("ssd_default_person")
        self.custom_nms = None

        # Initialize NMS if selected
        if nms_type == 'seq2seq-nms':
            self.custom_nms = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=0.8,
                                                app_feats='fmod', device=device)
            self.custom_nms.download(model_name='seq2seq_pets_jpd_fmod', path='.')
            self.custom_nms.load('./seq2seq_pets_jpd_fmod/', verbose=True)
            rospy.loginfo("Object Detection 2D SSD node seq2seq-nms initialized.")
        elif nms_type == 'soft-nms':
            self.custom_nms = SoftNMS(nms_thres=0.45, device=device)
            rospy.loginfo("Object Detection 2D SSD node soft-nms initialized.")
        elif nms_type == 'fast-nms':
            self.custom_nms = FastNMS(device=device)
            rospy.loginfo("Object Detection 2D SSD node fast-nms initialized.")
        elif nms_type == 'cluster-nms':
            self.custom_nms = ClusterNMS(device=device)
            rospy.loginfo("Object Detection 2D SSD node cluster-nms initialized.")
        else:
            rospy.loginfo("Object Detection 2D SSD node using default NMS.")

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_object_detection_2d_ssd_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Object detection 2D SSD node started.")
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
        boxes = self.object_detector.infer(image, threshold=0.45, keep_size=False, custom_nms=self.custom_nms)

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        # Publish detections in ROS message
        if self.object_publisher is not None:
            self.object_publisher.publish(self.bridge.to_ros_boxes(boxes))

        if self.image_publisher is not None:
            # Get an OpenCV image back
            image = image.opencv()
            # Annotate image with object detection boxes
            image = draw_bounding_boxes(image, boxes, class_names=self.object_detector.classes)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_objects_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/objects")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", help="Backbone network, defaults to vgg16_atrous",
                        type=str, default="vgg16_atrous", choices=["vgg16_atrous"])
    parser.add_argument("--nms_type", help="Non-Maximum Suppression type, defaults to \"default\", options are "
                                           "\"seq2seq-nms\", \"soft-nms\", \"fast-nms\", \"cluster-nms\"",
                        type=str, default="default",
                        choices=["default", "seq2seq-nms", "soft-nms", "fast-nms", "cluster-nms"])
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        if args.device == "cuda" and mx.context.num_gpus() > 0:
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

    object_detection_ssd_node = ObjectDetectionSSDNode(device=device, backbone=args.backbone, nms_type=args.nms_type,
                                                       input_rgb_image_topic=args.input_rgb_image_topic,
                                                       output_rgb_image_topic=args.output_rgb_image_topic,
                                                       detections_topic=args.detections_topic,
                                                       performance_topic=args.performance_topic)
    object_detection_ssd_node.listen()


if __name__ == '__main__':
    main()
