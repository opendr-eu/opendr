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

"""Camera publisher."""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class CameraPublisher(object):
    """Publish as a ROS topic the camera data."""

    def __init__(self, robot, jointPrefix, nodeName):
        self.robot = robot
        self.camera = robot.getDevice("panda_camera")
        self.camera.enable(50)
        self.img_publisher = rospy.Publisher('camera/color/raw', Image, queue_size=10)

    def publish(self):
        """Publish the 'joint_states' topic with up to date value."""
        cameraData = self.camera.getImage()
        h = self.camera.getHeight()
        w = self.camera.getWidth()

        image_msg = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'panda_camera'
        image_msg.height = h
        image_msg.width = w
        image_msg.encoding = 'bgra8'
        image_msg.data = cameraData
        image_msg.header = header
        image_msg.step = w * 4
        self.img_publisher.publish(image_msg)
