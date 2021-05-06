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

from opendr.engine.data import Image
from ros_bridge.msg import Pose as ROS_Pose
from opendr.engine.target import Pose
import numpy as np
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D


class ROSBridge:
    """
    This class provides an interface to convert OpenDR data types and targets into ROS-compatible ones similar to CvBridge.
    For each data type X two methods are provided:
    from_ros_X: which converts the ROS equivalent of X into OpenDR data type
    to_ros_X: which converts the OpenDR data type into the ROS equivalent of X
    """

    def __init__(self):
        self._cv_bridge = CvBridge()

    def from_ros_image(self, message, encoding='bgr8'):
        """
        Converts a ROS Image into an OpenDR image
        :param message: ROS image to be converted
        :type message: sensor_msgs.msg.Img
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: OpenDR image
        :rtype: engine.data.Image
        """
        cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        return image

    def to_ros_image(self, image, encoding='bgr8'):
        """
        Converts an OpenDR image into a ROS image
        :param image: OpenDR image to be converted
        :type image: engine.data.Image
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: ROS image
        :rtype: sensor_msgs.msg.Img
        """
        message = self._cv_bridge.cv2_to_imgmsg(image, encoding=encoding)
        return message

    def to_ros_pose(self, pose, image=None):
        """
        Converts an OpenDR pose into a Classification2D msg that can carry the same information
        Each keypoint is represented as a bbox centered at the keypoint with zero width/height. The unique subject id is
        also embedded on each keypoint (stored in ObjectHypothesisWithPose.id).
        :param pose: OpenDR pose to be converted
        :type pose: engine.target.Pose
        :param image: An image to embed into the message (optional)
        :type image: sensor_msgs.msg.Image
        :return: ROS message with the pose
        :rtype: vision_msgs.msg.Classification2D
        """
        ros_pose = Detection2D()
        if image is not None:
            ros_pose.source_img = image
        data = pose.data
        keypoints = Detection2DArray()
        for i in range(data.shape[0]):
            keypoint = Detection2D()
            keypoint.bbox = BoundingBox2D()
            keypoint.results.append(ObjectHypothesisWithPose())
            keypoint.bbox.center = Pose2D()
            keypoint.bbox.center.x = data[i][0]
            keypoint.bbox.center.y = data[i][1]
            keypoint.bbox.size_x = 0
            keypoint.bbox.size_y = 0
            keypoint.results[0].id = pose.id
            if pose.confidence:
                keypoint.results[0].score = pose.confidence
            keypoints.detections.append(keypoint)
        return keypoints

    def from_ros_pose(self, ros_pose):
        """
        Converts a ROS pose into an OpenDR pose
        :param ros_pose: ROS pose (as a payload to a vision_msgs/Classification2D) to be converted
        :type ros_pose: vision_msgs.msg.Classification2D
        :return: an OpenDR pose
        :rtype: engine.target.Pose
        """
        keypoints = ros_pose.detections
        data = []
        id = None
        confidence = None

        for keypoint in keypoints:
            data.append(keypoint.bbox.center.x)
            data.append(keypoint.bbox.center.y)
            confidence = keypoint.results[0].score
            id = keypoint.results[0].id
        data = np.asarray(data).reshape((-1, 2))

        pose = Pose(data, confidence)
        pose.id = id
        return pose
