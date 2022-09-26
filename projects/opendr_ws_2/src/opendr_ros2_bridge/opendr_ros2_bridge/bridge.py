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
from opendr.engine.data import Image, PointCloud
from opendr.engine.target import (
    Pose, BoundingBox, BoundingBoxList, Category,
    BoundingBox3D, BoundingBox3DList
)
from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image as ImageMsg, PointCloud as PointCloudMsg, ChannelFloat32 as ChannelFloat32Msg
from vision_msgs.msg import (
    Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose,
    Detection3D, Detection3DArray, BoundingBox3D as BoundingBox3DMsg
)
from geometry_msgs.msg import (
    Pose2D, Point32 as Point32Msg,
    Quaternion as QuaternionMsg, Pose as Pose3D
)
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

    def from_ros_point_cloud(self, point_cloud: PointCloudMsg):
        """
        Converts a ROS PointCloud message into an OpenDR PointCloud
        :param message: ROS PointCloud to be converted
        :type message: sensor_msgs.msg.PointCloud
        :return: OpenDR PointCloud
        :rtype: engine.data.PointCloud
        """

        points = np.empty([len(point_cloud.points), 3 + len(point_cloud.channels)], dtype=np.float32)

        for i in range(len(point_cloud.points)):
            point = point_cloud.points[i]
            x, y, z = point.x, point.y, point.z

            points[i, 0] = x
            points[i, 1] = y
            points[i, 2] = z

            for q in range(len(point_cloud.channels)):
                points[i, 3 + q] = point_cloud.channels[q].values[i]

        result = PointCloud(points)

        return result

    def to_ros_point_cloud(self, point_cloud, time_stamp):
        """
        Converts an OpenDR PointCloud message into a ROS2 PointCloud
        :param: OpenDR PointCloud
        :type: engine.data.PointCloud
        :return message: ROS PointCloud
        :rtype message: sensor_msgs.msg.PointCloud
        """

        ros_point_cloud = PointCloudMsg()

        header = Header()

        header.stamp = time_stamp
        ros_point_cloud.header = header

        channels_count = point_cloud.data.shape[-1] - 3

        channels = [ChannelFloat32Msg(name="channel_" + str(i), values=[]) for i in range(channels_count)]
        points = []

        for point in point_cloud.data:
            point_msg = Point32Msg()
            point_msg.x = float(point[0])
            point_msg.y = float(point[1])
            point_msg.z = float(point[2])
            points.append(point_msg)
            for i in range(channels_count):
                channels[i].values.append(float(point[3 + i]))

        ros_point_cloud.points = points
        ros_point_cloud.channels = channels

        return ros_point_cloud

    def from_ros_boxes_3d(self, ros_boxes_3d):
        """
        Converts a ROS2 Detection3DArray message into an OpenDR BoundingBox3D object.
        :param ros_boxes_3d: The ROS boxes to be converted.
        :type ros_boxes_3d: vision_msgs.msg.Detection3DArray
        :return: An OpenDR BoundingBox3DList object.
        :rtype: engine.target.BoundingBox3DList
        """
        boxes = []

        for ros_box in ros_boxes_3d:

            box = BoundingBox3D(
                name=ros_box.results[0].id,
                truncated=0,
                occluded=0,
                bbox2d=None,
                dimensions=np.array([
                    ros_box.bbox.size.position.x,
                    ros_box.bbox.size.position.y,
                    ros_box.bbox.size.position.z,
                ]),
                location=np.array([
                    ros_box.bbox.center.position.x,
                    ros_box.bbox.center.position.y,
                    ros_box.bbox.center.position.z,
                ]),
                rotation_y=ros_box.bbox.center.rotation.y,
                score=ros_box.results[0].score,
            )
            boxes.append(box)

        result = BoundingBox3DList(boxes)
        return result

    def to_ros_boxes_3d(self, boxes_3d):
        """
        Converts an OpenDR BoundingBox3DList object into a ROS2 Detection3DArray message.
        :param boxes_3d: The OpenDR boxes to be converted.
        :type boxes_3d: engine.target.BoundingBox3DList
        :param classes: The array of classes to transform from string name into an index.
        :return: ROS message with the boxes
        :rtype: vision_msgs.msg.Detection3DArray
        """
        ros_boxes_3d = Detection3DArray()
        for i in range(len(boxes_3d)):
            box = Detection3D()
            box.bbox = BoundingBox3DMsg()
            box.results.append(ObjectHypothesisWithPose())
            box.bbox.center = Pose3D()
            box.bbox.center.position.x = float(boxes_3d[i].location[0])
            box.bbox.center.position.y = float(boxes_3d[i].location[1])
            box.bbox.center.position.z = float(boxes_3d[i].location[2])
            box.bbox.center.orientation = QuaternionMsg(x=0.0, y=float(boxes_3d[i].rotation_y), z=0.0, w=0.0)
            box.bbox.size.x = float(boxes_3d[i].dimensions[0])
            box.bbox.size.y = float(boxes_3d[i].dimensions[1])
            box.bbox.size.z = float(boxes_3d[i].dimensions[2])
            box.results[0].id = boxes_3d[i].name
            box.results[0].score = float(boxes_3d[i].confidence)
            ros_boxes_3d.detections.append(box)
        return ros_boxes_3d
