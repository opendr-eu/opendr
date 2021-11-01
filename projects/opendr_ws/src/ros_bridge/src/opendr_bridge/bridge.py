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
from opendr.engine.target import Pose, BoundingBox, BoundingBoxList, Category

import cv2
import numpy as np
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose,\
     Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesis
from geometry_msgs.msg import Pose2D, Point, Pose as Pose3D
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image as ImageMsg


class ROSBridge:
    """
    This class provides an interface to convert OpenDR data types and targets into ROS-compatible ones similar to CvBridge.
    For each data type X two methods are provided:
    from_ros_X: which converts the ROS equivalent of X into OpenDR data type
    to_ros_X: which converts the OpenDR data type into the ROS equivalent of X
    """

    def __init__(self):
        self._cv_bridge = CvBridge()

    def from_ros_image(self, message: ImageMsg, encoding: str='passthrough') -> Image:
        """
        Converts a ROS image message into an OpenDR image
        :param message: ROS image to be converted
        :type message: sensor_msgs.msg.Image
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: OpenDR image (RGB)
        :rtype: engine.data.Image
        """
        cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        # Convert from OpenCV standard (BGR) to OpenDR standard (RGB)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        return image

    def to_ros_image(self, image: Image, encoding: str='passthrough') -> ImageMsg:
        """
        Converts an OpenDR image into a ROS image message
        :param image: OpenDR image (RGB) to be converted
        :type image: engine.data.Image
        :param encoding: encoding to be used for the conversion (inherited from CvBridge)
        :type encoding: str
        :return: ROS image
        :rtype: sensor_msgs.msg.Image
        """
        # Convert from the OpenDR standard (RGB) to OpenCV standard (BGR)
        bgr_image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        message = self._cv_bridge.cv2_to_imgmsg(bgr_image, encoding=encoding)
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

    def to_ros_face(self, category):
        """
        Converts an OpenDR category into a ObjectHypothesis msg that can carry the Category.data and Category.confidence.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS message with the category.data and category.confidence
        :rtype: vision_msgs.msg.ObjectHypothesis
        """
        result = ObjectHypothesis()
        result.id = category.data
        result.score = category.confidence
        return result

    def to_ros_face_id(self, category):
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

    def from_ros_face(self, ros_hypothesis):
        """
        Converts a ROS message with category payload into an OpenDR category
        :param ros_hypothesis: the objecthypothesis to be converted
        :type ros_face: vision_msgs.msg.ObjectHypothesis
        :return: an OpenDR category
        :rtype: engine.target.Category
        """
        category = Category(prediction=ros_hypothesis.id, description=None,
                            confidence=ros_hypothesis.score)
        return category

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
            detection.results[0].id = bounding_box.name
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

    def from_ros_3Dpose(self, ros_pose):
        """
        Converts a ROS message with pose payload into an OpenDR pose
        :param ros_pose: the pose to be converted (represented as vision_msgs.msg.Detection3DArray)
        :type ros_pose: vision_msgs.msg.Detection3DArray
        :return: an OpenDR pose
        :rtype: engine.target.Pose
        """
        keypoints = ros_pose.detections
        data = []
        pose_id, confidence = None, None

        for keypoint in keypoints:
            data.append(keypoint.bbox.center.position.x)
            data.append(keypoint.bbox.center.position.y)
            data.append(keypoint.bbox.center.position.z)
            confidence = keypoint.results[0].score
            pose_id = keypoint.results[0].id
        data = np.asarray(data).reshape((-1, 3))

        pose = Pose(data, confidence)
        pose.id = pose_id
        return pose

    def to_ros_3Dpose(self, pose):
        """
        Converts an OpenDR pose into a Detection3DArray msg that can carry the same information
        Each keypoint is represented as a bbox centered at the keypoint with zero radius. The subject id is also
        embedded on each keypoint (stored in ObjectHypothesisWithPose).
        :param pose: OpenDR pose to be converted
        :type pose: engine.target.Pose
        :return: ROS message with the pose
        :rtype: vision_msgs.msg.Detection3DArray
        """
        data = pose.data
        keypoints = Detection3DArray()
        for i in range(data.shape[0]):
            keypoint = Detection3D()
            keypoint.bbox = BoundingBox3D()
            keypoint.results.append(ObjectHypothesisWithPose())
            keypoint.bbox.center = Pose3D()
            keypoint.bbox.center.position.x = data[i, 0]
            keypoint.bbox.center.position.y = data[i, 1]
            keypoint.bbox.center.position.z = data[i, 2]
            keypoint.bbox.size.x = 0
            keypoint.bbox.size.y = 0
            keypoint.bbox.size.z = 0
            keypoint.results[0].id = int(pose.id)
            keypoint.results[0].score = 1
            keypoints.detections.append(keypoint)
        return keypoints

    def to_ros_mesh(self, vertices, faces):
        """
        Converts a mesh into a ROS Mesh
        :param vertices: the vertices of the 3D model
        :type vertices: numpy array (Nx3)
        :param faces: the faces of the 3D model
        :type faces: numpy array (Nx3)
        :return mesh_ROS: a ROS mesh
        :rtype mesh_ROS: shape_msgs.msg.Mesh
        """
        mesh_ROS = Mesh()
        for i in range(vertices.shape[0]):
            point = Point(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            mesh_ROS.vertices.append(point)
        for i in range(faces.shape[0]):
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices[0] = int(faces[i][0])
            mesh_triangle.vertex_indices[1] = int(faces[i][1])
            mesh_triangle.vertex_indices[2] = int(faces[i][2])
            mesh_ROS.triangles.append(mesh_triangle)
        return mesh_ROS

    def to_ros_colors(self, colors):
        """
        Converts an array of vertex_colors to a list of ROS colors
        :type colors: numpy array (Nx3)
        :return colors: a list of the colors of the vertices
        :rtype ros_colors: std_msgs.msg.ColorRGBA[]
        """
        ros_colors = []
        for i in range(colors.shape[0]):
            color = ColorRGBA(colors[i, 0], colors[i, 1], colors[i, 2], 0)
            ros_colors.append(color)
        return ros_colors

    def from_ros_mesh(self, mesh_ROS, vertex_colors_ROS=None):
        """
        Converts a ROS mesh into arrays of vertices and faces of a mesh
        :param mesh_ROS: the ROS mesh to be converted
        :type mesh_ROS: shape_msgs.msg.Mesh
        :param vertex_colors_ROS: a list of the colors of the vertices
        :type vertex_colors_ROS: std_msgs.msg.ColorRGBA[]
        :return vertices: the vertices of the 3D model
        :rtype vertices: numpy array (Nx3)
        :return faces: the faces of the 3D model
        :rtype faces: numpy array (Nx3)
        :return vertex_colors: the colors of the vertices of the 3D model
        :rtype vertex_colors: numpy array (Nx3)
        """
        vertices = np.zeros([len(mesh_ROS.vertices), 3])
        faces = np.zeros([len(mesh_ROS.triangles), 3]).astype(int)
        for i in range(len(mesh_ROS.vertices)):
            vertices[i] = np.array([mesh_ROS.vertices[i].x, mesh_ROS.vertices[i].y, mesh_ROS.vertices[i].z])
        for i in range(len(mesh_ROS.triangles)):
            faces[i] = np.array([int(mesh_ROS.triangles[i].vertex_indices[0]), int(mesh_ROS.triangles[i].vertex_indices[1]),
                                 int(mesh_ROS.triangles[i].vertex_indices[2])]).astype(int)
        return vertices, faces

    def from_ros_colors(self, ros_colors):
        """
        Converts a list of ROS colors into a list of colors
        :param ros_colors: a list of the colors of the vertices
        :type ros_colors: std_msgs.msg.ColorRGBA[]
        :return colors: the colors of the vertices of the 3D model
        :rtype colors: numpy array (Nx3)
        """
        colors = np.zeros([len(ros_colors), 3])
        for i in range(len(ros_colors)):
            colors[i] = np.array([ros_colors[i].r, ros_colors[i].g, ros_colors[i].b])
        return colors
