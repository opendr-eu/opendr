# Copyright 2020-2022 OpenDR European Project
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

import numpy as np
from opendr.engine.data import Image
from opendr.engine.target import Pose, BoundingBox, BoundingBoxList

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D


class ROS2Bridge:

    def __init__(self):
        self._cv_bridge = CvBridge()

    def from_ros_image(self, message: ImageMsg, encoding: str='passthrough') -> Image:
        # Commented code converts to opencv format without the use of cvbridge
        # sz = (message.height, message.width)
        # # print(msg.header.stamp)
        #
        # # print("{encoding} {width} {height} {step} {data_size}".format(
        # #     encoding=msg.encoding, width=msg.width, height=msg.height,
        # #     step=msg.step, data_size=len(msg.data)))
        # if message.step * message.height != len(message.data):
        #     print("bad step/height/data size")
        #     return
        #
        # if message.encoding == "rgb8":
        #     img = np.zeros([message.height, message.width, 3], dtype=np.uint8)
        #     img[:, :, 2] = np.array(message.data[0::3]).reshape(sz)
        #     img[:, :, 1] = np.array(message.data[1::3]).reshape(sz)
        #     img[:, :, 0] = np.array(message.data[2::3]).reshape(sz)
        # elif message.encoding == "mono8":
        #     img = np.array(message.data).reshape(sz)
        # else:
        #     print("unsupported encoding {}".format(message.encoding))
        #     return
        #
        # # Convert image from OpenCV format to OpenDR format
        # return Image(np.asarray(img, dtype=np.uint8))

        cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        return image

    def to_ros_image(self, image: Image, encoding: str='passthrough') -> ImageMsg:
        # Convert from the OpenDR standard (CHW/RGB) to OpenCV standard (HWC/BGR)
        message = self._cv_bridge.cv2_to_imgmsg(image.opencv(), encoding=encoding)
        return message

    def to_ros_pose(self, pose):
        data = pose.data
        keypoints = Detection2DArray()
        for i in range(data.shape[0]):
            keypoint = Detection2D()
            keypoint.bbox = BoundingBox2D()
            keypoint.results.append(ObjectHypothesisWithPose())
            keypoint.bbox.center = Pose2D()
            keypoint.bbox.center.x = float(data[i][0])
            keypoint.bbox.center.y = float(data[i][1])
            keypoint.bbox.size_x = 0.0
            keypoint.bbox.size_y = 0.0
            keypoint.results[0].id = str(pose.id)
            if pose.confidence:
                keypoint.results[0].score = pose.confidence
            keypoints.detections.append(keypoint)
        return keypoints

    def from_ros_pose(self, ros_pose):
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


    def to_ros_bounding_box_list(self, bounding_box_list):
        """
        Converts an OpenDR bounding_box_list into a Detection2DArray msg that can carry the same information
        The object class is also embedded on each bounding box (stored in ObjectHypothesisWithPose).
        :param bounding_box_list: OpenDR bounding_box_list to be converted
        :type bounding_box_list: engine.target.BoundingBoxList
        :return: ROS message with the bounding box list
        :rtype: vision_msgs.msg.Detection2DArray
        """
        detections = Detection2DArray()
        for bounding_box in bounding_box_list:
            detection = Detection2D()
            detection.bbox = BoundingBox2D()
            detection.results.append(ObjectHypothesisWithPose())
            detection.bbox.center = Pose2D()
            detection.bbox.center.x = bounding_box.left + bounding_box.width / 2.0
            detection.bbox.center.y = bounding_box.top + bounding_box.height / 2.0
            detection.bbox.size_x = bounding_box.width
            detection.bbox.size_y = bounding_box.height
            detection.results[0].id = str(bounding_box.name)
            detection.results[0].score = bounding_box.confidence
            detections.detections.append(detection)
        return detections


    def from_ros_bounding_box_list(self, ros_detection_2d_array):
        """
        Converts a ROS message with bounding box list payload into an OpenDR pose
        :param ros_detection_2d_array: the bounding boxes to be converted (represented as vision_msgs.msg.Detection2DArray)
        :type ros_detection_2d_array: vision_msgs.msg.Detection2DArray
        :return: an OpenDR bounding box list
        :rtype: engine.target.BoundingBoxList
        """
        detections = ros_detection_2d_array.detections
        boxes = []

        for detection in detections:
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            left = detection.bbox.center.x - width / 2.0
            top = detection.bbox.center.y - height / 2.0
            name = detection.results[0].id
            score = detection.results[0].confidence
            boxes.append(BoundingBox(name, left, top, width, height, score))
        bounding_box_list = BoundingBoxList(boxes)
        return bounding_box_list