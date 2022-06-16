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

    def to_ros_boxes(self, box_list):
        boxes = box_list.data
        ros_boxes = Detection2DArray()
        for idx, box in enumerate(boxes):
            ros_box = Detection2D()
            ros_box.bbox = BoundingBox2D()
            ros_box.results.append(ObjectHypothesisWithPose())
            ros_box.bbox.center = Pose2D()
            ros_box.bbox.center.x = box.left + box.width / 2.
            ros_box.bbox.center.y = box.top + box.height / 2.
            ros_box.bbox.size_x = float(box.width)
            ros_box.bbox.size_y = float(box.height)
            ros_box.results[0].id = str(box.name)
            if box.confidence:
                ros_box.results[0].score = float(box.confidence)
            ros_boxes.detections.append(ros_box)
        return ros_boxes

    def from_ros_boxes(self, ros_detections):
        ros_boxes = ros_detections.detections
        bboxes = BoundingBoxList(boxes=[])

        for idx, box in enumerate(ros_boxes):
            width = box.bbox.size_x
            height = box.bbox.size_y
            left = box.bbox.center.x - width / 2.
            top = box.bbox.center.y - height / 2.
            # _id = box.results[0].id[0]
            _id = int(float(box.results[0].id.strip('][').split(', ')[0]))
            bbox = BoundingBox(top=top, left=left, width=width, height=height, name=_id)
            bboxes.data.append(bbox)

        return bboxes

    def to_ros_bounding_box_list(self, bounding_box_list):
        detections = Detection2DArray()
        for bounding_box in bounding_box_list:
            detection = Detection2D()
            detection.bbox = BoundingBox2D()
            detection.results.append(ObjectHypothesisWithPose())
            detection.bbox.center = Pose2D()
            detection.bbox.center.x = bounding_box.left + bounding_box.width / 2.0
            detection.bbox.center.y = bounding_box.top + bounding_box.height / 2.0
            detection.bbox.size_x = float(bounding_box.width)
            detection.bbox.size_y = float(bounding_box.height)
            detection.results[0].id = str(bounding_box.name)
            detection.results[0].score = float(bounding_box.confidence)
            detections.detections.append(detection)
        return detections

    def from_ros_bounding_box_list(self, ros_detection_2d_array):
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
