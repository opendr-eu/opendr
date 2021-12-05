# Copyright 1996-2020 Cyberbotics Ltd.
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

"""Camera publisher."""

import rospy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header

from std_msgs.msg import Float64
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError



class CameraPublisher(object):
    """Publish as a ROS topic the camera data."""


    def __init__(self, robot, jointPrefix, nodeName):
        self.robot = robot
        self.camera = robot.getDevice("panda_camera")
        self.camera.enable(50)
        self.img_publisher = rospy.Publisher('camera/color/raw', Image,
                                                            queue_size=10)
        self.focal_length_publisher = rospy.Publisher('camera/focal_length', Float64,
                                                            queue_size=1)
    def publish(self):
        """Publish the 'joint_states' topic with up to date value."""


        cameraData = self.camera.getImage()
        focal = self.camera.getFocalLength()
        h = self.camera.getHeight()
        w = self.camera.getWidth()

        image_msg=Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'panda_camera'
        image_msg.height = h
        image_msg.width = w
        image_msg.encoding ='bgra8'
        image_msg.data = cameraData
        image_msg.header = header
        image_msg.step= w * 4
        self.img_publisher.publish(image_msg)
