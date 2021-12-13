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

import torch
import rospy
import time
import numpy as np
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.engine.datasets import DatasetIterator
from opendr.engine.data import Image
from opendr.perception.object_tracking_2d.datasets.mot_dataset import RawMotDatasetIterator

class ImageDatasetNode:
    def __init__(
        self,
        dataset: DatasetIterator,
        output_image_topic="/opendr/dataset_image",
    ):
        """
        Creates a ROS Node for publishing dataset images
        """

        # Initialize the face detector
        self.dataset = dataset
        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        if output_image_topic is not None:
            self.output_image_publisher = rospy.Publisher(
                output_image_topic, ROS_Image, queue_size=10
            )

    def start(self):
        rospy.loginfo("Timing images")

        i = 0

        while not rospy.is_shutdown():

            image: Image = self.dataset[i % len(self.dataset)][0]  # Dataset should have an (Image, Target) pair as elements

            print(image.data.shape)
            # image = Image(np.ones([500, 30, 3], dtype=np.uint8) * (i % 22) * 10)

            rospy.loginfo('Publishing image [' + str(i) + ']')
            message = self.bridge.to_ros_image(
                image, encoding="rgb8"
            )
            self.output_image_publisher.publish(message)

            time.sleep(0.1)

            i += 1


if __name__ == "__main__":
    
    rospy.init_node('opendr_image_dataset')
    dataset = RawMotDatasetIterator(
        "/mnt/e/FILES/MOT/MOT2020",
        {
            "nano_mot20": "/home/io/opendr_internal/src/opendr/perception/object_tracking_2d/datasets/splits/mot20.train"
        },
        scan_labels=False
    )
    dataset_node = ImageDatasetNode(dataset)
    dataset_node.start()

