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
from opendr.engine.target import Pose, BoundingBoxList, BoundingBox
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

    def to_ros_pose(self, pose):
        """
        Converts an OpenDR pose into a Detection2DArray msg that can carry the same information
        Each keypoint is represented as a bbox centered at the keypoint with zero width/height. The subject id is also
        embedded on each keypoint (stored in ObjectHypothesisWithPose).
        :param pose: OpenDR pose to be converted
        :type pose: engine.target.Pose
        :return: ROS message with the pose
        :rtype: vision_msgs.msg.Detection2DArray
        """
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
        Converts a ROS message with pose payload into an OpenDR pose
        :param ros_pose: the pose to be converted (represented as vision_msgs.msg.Detection2DArray)
        :type ros_pose: vision_msgs.msg.Detection2DArray
        :return: an OpenDR pose
        :rtype: engine.target.Pose
        """
        keypoints = ros_pose.detections
        data = []
        pose_id, confidence = None, None

        for keypoint in keypoints:
            data.append(keypoint.bbox.center.x)
            data.append(keypoint.bbox.center.y)
            confidence = keypoint.results[0].score
            pose_id = keypoint.results[0].id
        data = np.asarray(data).reshape((-1, 2))

        pose = Pose(data, confidence)
        pose.id = pose_id
        return pose

    def to_ros_boxes(self, box_list):
        """
        Converts an OpenDR BoundingBoxList into a Detection2DArray msg that can carry the same information.
        Each bounding box is represented by its center coordinates as well as its width/height dimensions.
        :param box_list: OpenDR bounding boxes to be converted
        :type box_list: engine.target.BoundingBoxList
        :return: ROS message with the bounding boxes
        :rtype: vision_msgs.msg.Detection2DArray
        """
        boxes = box_list.data
        ros_boxes = Detection2DArray()
        for idx, box in enumerate(boxes):
            ros_box = Detection2D()
            ros_box.bbox = BoundingBox2D()
            ros_box.results.append(ObjectHypothesisWithPose())
            ros_box.bbox.center = Pose2D()
            ros_box.bbox.center.x = box.left + box.width / 2.
            ros_box.bbox.center.y = box.top + box.height / 2.
            ros_box.bbox.size_x = box.width
            ros_box.bbox.size_y = box.height
            ros_box.results[0].id = box.name
            if box.confidence:
                ros_box.results[0].score = box.confidence
            ros_boxes.detections.append(ros_box)
        return ros_boxes

    def from_ros_boxes(self, ros_detections):
        """
        Converts a ROS message with bounding boxes into an OpenDR BoundingBoxList
        :param ros_detections: the boxes to be converted (represented as vision_msgs.msg.Detection2DArray)
        :type ros_detections: vision_msgs.msg.Detection2DArray
        :return: an OpenDR BoundingBoxList
        :rtype: engine.target.BoundingBoxList
        """
        ros_boxes = ros_detections.detections
        bboxes = BoundingBoxList(boxes=[])

        for idx, box in enumerate(ros_boxes):
            width = box.bbox.size_x
            height = box.bbox.size_y
            left = box.bbox.center.x - width / 2.
            top = box.bbox.center.y - height / 2.
            id = box.results[0].id
            bbox = BoundingBox(top=top, left=left, width=width, height=height, name=id)
            bboxes.data.append(bbox)

        return bboxes
