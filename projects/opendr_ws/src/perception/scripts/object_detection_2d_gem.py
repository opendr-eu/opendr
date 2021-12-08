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
import message_filters
import cv2
import time
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.object_detection_2d import GemLearner
from opendr.perception.object_detection_2d import draw
from opendr.engine.data import Image


class GemNode:

    def __init__(self,
                 input_color_topic="/camera/color/image_raw",
                 input_infra_topic="/camera/infra/image_raw",
                 output_color_topic="/opendr/color_detection_annotated",
                 output_infra_topic="/opendr/infra_detection_annotated",
                 detection_annotations_topic="/opendr/detections",
                 device="cuda",
                 pts_color=None,
                 pts_infra=None,
                 ):
        """
        Creates a ROS Node for object detection with GEM
        :param input_color_topic: Topic from which we are reading the input color image
        :type input_color_topic: str
        :param input_infra_topic: Topic from which we are reading the input infrared image
        :type: input_infra_topic: str
        :param output_color_topic: Topic to which we are publishing the annotated color image (if None, we are not
        publishing annotated image)
        :type output_color_topic: str
        :param output_infra_topic: Topic to which we are publishing the annotated infrared image (if None, we are not
        publishing annotated image)
        :type output_infra_topic: str
        :param detection_annotations_topic: Topic to which we are publishing the annotations (if None, we are
        not publishing annotations)
        :type detection_annotations_topic:  str
        :param device: Device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param pts_color: Point on the color image that define alignment with the infrared image. These are camera
        specific and can be obtained using get_color_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_color: {list, numpy.ndarray}
        :param pts_infra: Points on the infrared image that define alignment with color image. These are camera specific
        and can be obtained using get_color_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_infra: {list, numpy.ndarray}
        """
        rospy.init_node('gem', anonymous=True)
        if output_color_topic is not None:
            self.rgb_publisher = rospy.Publisher(output_color_topic, ROS_Image, queue_size=10)
        else:
            self.rgb_publisher = None
        if output_infra_topic is not None:
            self.ir_publisher = rospy.Publisher(output_infra_topic, ROS_Image, queue_size=10)
        else:
            self.ir_publisher = None

        if detection_annotations_topic is not None:
            self.detection_publisher = rospy.Publisher(detection_annotations_topic, Detection2DArray, queue_size=10)
        else:
            self.detection_publisher = None
        if pts_infra is None:
            pts_infra = np.array([[478, 248], [465, 338], [458, 325], [468, 256],
                                  [341, 240], [335, 310], [324, 321], [311, 383],
                                  [434, 365], [135, 384], [67, 257], [167, 206],
                                  [124, 131], [364, 276], [424, 269], [277, 131],
                                  [41, 310], [202, 320], [188, 318], [188, 308],
                                  [196, 241], [499, 317], [311, 164], [220, 216],
                                  [435, 352], [213, 363], [390, 364], [212, 368],
                                  [390, 370], [467, 324], [415, 364]])
            rospy.logwarn(
                '\nUsing default calibration values for pts_infra!' +
                '\nThese are probably incorrect.' +
                '\nThe correct values for pts_infra can be found by running get_color_infra_alignment.py.' +
                '\nThis file is located in the opendr/perception/object_detection2d/utils module.'
            )
        if pts_color is None:
            pts_color = np.array([[910, 397], [889, 572], [874, 552], [891, 411],
                                  [635, 385], [619, 525], [603, 544], [576, 682],
                                  [810, 619], [216, 688], [90, 423], [281, 310],
                                  [193, 163], [684, 449], [806, 431], [504, 170],
                                  [24, 538], [353, 552], [323, 550], [323, 529],
                                  [344, 387], [961, 533], [570, 233], [392, 336],
                                  [831, 610], [378, 638], [742, 630], [378, 648],
                                  [742, 640], [895, 550], [787, 630]])
            rospy.logwarn(
                '\nUsing default calibration values for pts_color!' +
                '\nThese are probably incorrect.' +
                '\nThe correct values for pts_color can be found by running get_color_infra_alignment.py.' +
                '\nThis file is located in the opendr/perception/object_detection2d/utils module.'
            )
        # Object classes
        self.classes = ['N/A', 'chair', 'cycle', 'bin', 'laptop', 'drill', 'rocker']

        # Estimating Homography matrix for aligning infra with RGB
        self.h, status = cv2.findHomography(pts_infra, pts_color)

        self.bridge = ROSBridge()

        # Initialize the detection estimation
        model_backbone = "resnet50"

        self.gem_learner = GemLearner(backbone=model_backbone,
                                      num_classes=7,
                                      device=device,
                                      )
        self.gem_learner.fusion_method = 'sc_avg'
        self.gem_learner.download(path=".", verbose=True)

        # Subscribers
        msg_rgb = message_filters.Subscriber(input_color_topic, ROS_Image)
        msg_ir = message_filters.Subscriber(input_infra_topic, ROS_Image)

        sync = message_filters.TimeSynchronizer([msg_rgb, msg_ir], 1)
        sync.registerCallback(self.callback)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        self.fps_list = []
        rospy.loginfo("GEM node started!")
        rospy.spin()

    def callback(self, msg_rgb, msg_ir):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param msg_rgb: input color image message
        :type msg_rgb: sensor_msgs.msg.Image
        :param msg_ir: input infrared image message
        :type msg_ir: sensor_msgs.msg.Image
        """
        # Convert images to OpenDR standard
        image_rgb = self.bridge.from_ros_image(msg_rgb).opencv()
        image_ir_raw = self.bridge.from_ros_image(msg_ir).opencv()
        image_ir = cv2.warpPerspective(image_ir_raw, self.h, (image_rgb.shape[1], image_rgb.shape[0]))

        # Perform inference on images
        start = time.time()
        boxes, w_sensor1, _ = self.gem_learner.infer(image_rgb, image_ir)
        end = time.time()

        # Calculate fps
        fps = 1 / (end - start)
        self.fps_list.append(fps)
        if len(self.fps_list) > 10:
            del self.fps_list[0]
        mean_fps = sum(self.fps_list) / len(self.fps_list)

        #  Annotate image and publish results:
        if self.detection_publisher is not None:
            ros_detection = self.bridge.to_ros_bounding_box_list(boxes)
            self.detection_publisher.publish(ros_detection)
            # We get can the data back using self.bridge.from_ros_bounding_box_list(ros_detection)
            # e.g., opendr_detection = self.bridge.from_ros_bounding_box_list(ros_detection)

        if self.rgb_publisher is not None:
            plot_rgb = draw(image_rgb, boxes, w_sensor1, mean_fps)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_rgb)))
            self.rgb_publisher.publish(message)
        if self.ir_publisher is not None:
            plot_ir = draw(image_ir, boxes, w_sensor1, mean_fps)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_ir)))
            self.ir_publisher.publish(message)


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
    detection_estimation_node = GemNode(device=device)
    detection_estimation_node.listen()
