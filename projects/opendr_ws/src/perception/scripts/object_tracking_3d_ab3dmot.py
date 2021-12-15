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

import os
import torch
from opendr.engine.learners import Learner
import rospy
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROSBridge
from opendr.perception.object_tracking_3d import ObjectTracking3DAb3dmotLearner
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner


class ObjectTracking3DAb3dmotNode:
    def __init__(
        self,
        detector: Learner,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        output_detection3d_topic="/opendr/detection3d",
        output_tracking3d_id_topic="/opendr/tracking3d_id",
        device="cuda:0",
    ):
        """
        Creates a ROS Node for 3D object tracking
        :param detector: Learner that proides 3D object detections
        :type detector: Learner
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_image_topic: str
        :param output_detection3d_topic: Topic to which we are publishing the annotations
        :type output_detection3d_topic:  str
        :param output_tracking3d_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking3d_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        self.detector = detector
        self.learner = ObjectTracking3DAb3dmotLearner(
            device=device
        )

        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        self.detection_publisher = rospy.Publisher(
            output_detection3d_topic, Detection3DArray, queue_size=10
        )
        self.tracking_id_publisher = rospy.Publisher(
            output_tracking3d_id_topic, Int32MultiArray, queue_size=10
        )

        rospy.Subscriber(input_point_cloud_topic, ROS_PointCloud, self.callback)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        point_cloud = self.bridge.from_ros_point_cloud(data)
        detection_boxes = self.detector.infer(point_cloud)
        tracking_boxes = self.learner.infer(detection_boxes)
        ids = [tracking_box.id for tracking_box in tracking_boxes]

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes_3d(detection_boxes, classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
        if self.detection_publisher is not None:
            self.detection_publisher.publish(ros_boxes)
            rospy.loginfo("Published detection boxes")

        ros_ids = Int32MultiArray()
        ros_ids.data = ids

        if self.tracking_id_publisher is not None:
            self.tracking_id_publisher.publish(ros_ids)
            rospy.loginfo("Published tracking ids")

if __name__ == "__main__":
    # Automatically run on GPU/CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # initialize ROS node
    rospy.init_node("opendr_voxel_detection_3d", anonymous=True)
    rospy.loginfo("AB3DMOT node started")

    input_point_cloud_topic = rospy.get_param(
        "~input_point_cloud_topic", "/opendr/dataset_point_cloud"
    )
    temp_dir = rospy.get_param("~temp_dir", "temp")
    detector_model_name = rospy.get_param("~detector_model_name", "tanet_car_xyres_16")
    detector_model_config_path = rospy.get_param(
        "~detector_model_config_path", os.path.join(
            "..", "..", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "car", "test_short.proto"
        )
    )

    detector = VoxelObjectDetection3DLearner(
        device=device, temp_path=temp_dir, model_config_path=detector_model_config_path
    )
    if not os.path.exists(os.path.join(temp_dir, detector_model_name)):
        VoxelObjectDetection3DLearner.download(detector_model_name, temp_dir)

    detector.load(os.path.join(temp_dir, detector_model_name), verbose=True)

    # created node object
    ab3dmot_node = ObjectTracking3DAb3dmotNode(
        detector=detector,
        device=device,
        input_point_cloud_topic=input_point_cloud_topic,
    )
    # begin ROS communications
    rospy.spin()
