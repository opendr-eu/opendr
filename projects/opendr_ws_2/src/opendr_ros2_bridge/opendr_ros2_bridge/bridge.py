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
from opendr.engine.data import Image, Timeseries
from opendr.engine.target import Pose, BoundingBox, BoundingBoxList, Category

from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image as ImageMsg
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose, \
    Classification2D
from geometry_msgs.msg import Pose2D
from opendr_ros2_messages.msg import OpenDRPose2D, OpenDRPose2DKeypoint


class ROS2Bridge:
    """
    This class provides an interface to convert OpenDR data types and targets into ROS2-compatible ones similar
    to CvBridge.
    For each data type X two methods are provided:
    from_ros_X: which converts the ROS2 equivalent of X into OpenDR data type
    to_ros_X: which converts the OpenDR data type into the ROS2 equivalent of X
    """

    def __init__(self):
        self._cv_bridge = CvBridge()

    def to_ros_image(self, image: Image, encoding: str='passthrough') -> ImageMsg:
        """
        Converts an OpenDR image into a ROS2 image message
        :param image: OpenDR image to be converted
        :type image: engine.data.Image
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: ROS2 image
        :rtype: sensor_msgs.msg.Image
        """
        # Convert from the OpenDR standard (CHW/RGB) to OpenCV standard (HWC/BGR)
        message = self._cv_bridge.cv2_to_imgmsg(image.opencv(), encoding=encoding)
        return message

    def from_ros_image(self, message: ImageMsg, encoding: str='passthrough') -> Image:
        """
        Converts a ROS2 image message into an OpenDR image
        :param message: ROS2 image to be converted
        :type message: sensor_msgs.msg.Image
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: OpenDR image (RGB)
        :rtype: engine.data.Image
        """
        cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        return image

    def to_ros_pose(self, pose: Pose):
        """
        Converts an OpenDR Pose into a OpenDRPose2D msg that can carry the same information, i.e. a list of keypoints,
        the pose detection confidence and the pose id.
        Each keypoint is represented as an OpenDRPose2DKeypoint with x, y pixel position on input image with (0, 0)
        being the top-left corner.
        :param pose: OpenDR Pose to be converted to OpenDRPose2D
        :type pose: engine.target.Pose
        :return: ROS message with the pose
        :rtype: opendr_ros2_messages.msg.OpenDRPose2D
        """
        data = pose.data
        # Setup ros pose
        ros_pose = OpenDRPose2D()
        ros_pose.pose_id = int(pose.id)
        if pose.confidence:
            ros_pose.conf = pose.confidence

        # Add keypoints to pose
        for i in range(data.shape[0]):
            ros_keypoint = OpenDRPose2DKeypoint()
            ros_keypoint.kpt_name = pose.kpt_names[i]
            ros_keypoint.x = int(data[i][0])
            ros_keypoint.y = int(data[i][1])
            # Add keypoint to pose
            ros_pose.keypoint_list.append(ros_keypoint)
        return ros_pose

    def from_ros_pose(self, ros_pose: OpenDRPose2D):
        """
        Converts an OpenDRPose2D message into an OpenDR Pose.
        :param ros_pose: the ROS pose to be converted
        :type ros_pose: opendr_ros2_messages.msg.OpenDRPose2D
        :return: an OpenDR Pose
        :rtype: engine.target.Pose
        """
        ros_keypoints = ros_pose.keypoint_list
        keypoints = []
        pose_id, confidence = ros_pose.pose_id, ros_pose.conf

        for ros_keypoint in ros_keypoints:
            keypoints.append(int(ros_keypoint.x))
            keypoints.append(int(ros_keypoint.y))
        data = np.asarray(keypoints).reshape((-1, 2))

        pose = Pose(data, confidence)
        pose.id = pose_id
        return pose

    def to_ros_boxes(self, box_list):
        """
        Converts an OpenDR BoundingBoxList into a Detection2DArray msg that can carry the same information.
        Each bounding box is represented by its center coordinates as well as its width/height dimensions.
        :param box_list: OpenDR bounding boxes to be converted
        :type box_list: engine.target.BoundingBoxList
        :return: ROS2 message with the bounding boxes
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
            ros_box.bbox.size_x = float(box.width)
            ros_box.bbox.size_y = float(box.height)
            ros_box.results[0].id = str(box.name)
            if box.confidence:
                ros_box.results[0].score = float(box.confidence)
            ros_boxes.detections.append(ros_box)
        return ros_boxes

    def from_ros_boxes(self, ros_detections):
        """
        Converts a ROS2 message with bounding boxes into an OpenDR BoundingBoxList
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
            _id = int(float(box.results[0].id.strip('][').split(', ')[0]))
            bbox = BoundingBox(top=top, left=left, width=width, height=height, name=_id)
            bboxes.data.append(bbox)
        return bboxes

    def to_ros_bounding_box_list(self, bounding_box_list):
        """
        Converts an OpenDR bounding_box_list into a Detection2DArray msg that can carry the same information
        The object class is also embedded on each bounding box (stored in ObjectHypothesisWithPose).
        :param bounding_box_list: OpenDR bounding_box_list to be converted
        :type bounding_box_list: engine.target.BoundingBoxList
        :return: ROS2 message with the bounding box list
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
            detection.bbox.size_x = float(bounding_box.width)
            detection.bbox.size_y = float(bounding_box.height)
            detection.results[0].id = str(bounding_box.name)
            detection.results[0].score = float(bounding_box.confidence)
            detections.detections.append(detection)
        return detections

    def from_ros_bounding_box_list(self, ros_detection_2d_array):
        """
        Converts a ROS2 message with bounding box list payload into an OpenDR pose
        :param ros_detection_2d_array: the bounding boxes to be converted (represented as
                                       vision_msgs.msg.Detection2DArray)
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

    def to_ros_face(self, category):
        """
        Converts an OpenDR category into a ObjectHypothesis msg that can carry the Category.data and
        Category.confidence.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS2 message with the category.data and category.confidence
        :rtype: vision_msgs.msg.ObjectHypothesis
        """
        result = ObjectHypothesisWithPose()
        result.id = str(category.data)
        result.score = category.confidence
        return result

    def from_ros_face(self, ros_hypothesis):
        """
        Converts a ROS2 message with category payload into an OpenDR category
        :param ros_hypothesis: the object hypothesis to be converted
        :type ros_hypothesis: vision_msgs.msg.ObjectHypothesis
        :return: an OpenDR category
        :rtype: engine.target.Category
        """
        return Category(prediction=ros_hypothesis.id, description=None,
                        confidence=ros_hypothesis.score)

    def to_ros_face_id(self, category):
        """
        Converts an OpenDR category into a string msg that can carry the Category.description.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS2 message with the category.description
        :rtype: std_msgs.msg.String
        """
        result = String()
        result.data = category.description
        return result

    def to_ros_category(self, category):
        """
        Converts an OpenDR category into a ObjectHypothesis msg that can carry the Category.data and Category.confidence.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS message with the category.data and category.confidence
        :rtype: vision_msgs.msg.ObjectHypothesis
        """
        result = ObjectHypothesis()
        result.id = str(category.data)
        result.score = category.confidence
        return result

    def from_ros_category(self, ros_hypothesis):
        """
        Converts a ROS message with category payload into an OpenDR category
        :param ros_hypothesis: the object hypothesis to be converted
        :type ros_hypothesis: vision_msgs.msg.ObjectHypothesis
        :return: an OpenDR category
        :rtype: engine.target.Category
        """
        category = Category(prediction=ros_hypothesis.id, description=None,
                            confidence=ros_hypothesis.score)
        return category

    def to_ros_category_description(self, category):
        """
        Converts an OpenDR category into a string msg that can carry the Category.description.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS message with the category.description
        :rtype: std_msgs.msg.String
        """
        result = String()
        result.data = category.description
        return result

    def from_rosarray_to_timeseries(self, ros_array, dim1, dim2):
        """
        Converts ROS2 array into OpenDR Timeseries object
        :param ros_array: data to be converted
        :type ros_array: std_msgs.msg.Float32MultiArray
        :param dim1: 1st dimension
        :type dim1: int
        :param dim2: 2nd dimension
        :type dim2: int
        :rtype: engine.data.Timeseries
        """
        data = np.reshape(ros_array.data, (dim1, dim2))
        data = Timeseries(data)
        return data

    def from_category_to_rosclass(self, prediction, timestamp, source_data=None):
        """
        Converts OpenDR Category into Classification2D message with class label, confidence, timestamp and corresponding input
        :param prediction: classification prediction
        :type prediction: engine.target.Category
        :param timestamp: time stamp for header message
        :type timestamp: str
        :param source_data: corresponding input or None
        :return classification
        :rtype: vision_msgs.msg.Classification2D
        """
        classification = Classification2D()
        classification.header = Header()
        classification.header.stamp = timestamp

        result = ObjectHypothesis()
        result.id = str(prediction.data)
        result.score = prediction.confidence
        classification.results.append(result)
        if source_data is not None:
            classification.source_img = source_data
        return classification
