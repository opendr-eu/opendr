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
from opendr.engine.target import Pose, BoundingBox, BoundingBoxList
import numpy as np
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose,\
     Detection3DArray, Detection3D, BoundingBox3D
from geometry_msgs.msg import Pose2D, Point, Pose as Pose3D
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import ColorRGBA


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

    def to_ros_mesh(self, vertices, faces, vertex_colors=None):
        """
        Converts a mesh into a ROS Mesh and and to a list of vertex_colors
        :param vertices: the vertices of the 3D model
        :type vertices: numpy array (Nx3)
        :param faces: the faces of the 3D model
        :type faces: numpy array (Nx3)
        :param vertex_colors: the colors of the vertices of the 3D model
        :type vertex_colors: numpy array (Nx3)
        :return mesh_ROS: a ROS mesh
        :rtype mesh_ROS: shape_msgs.msg.Mesh
        :return vertex_colors_ROS: a list of the colors of the vertices
        :rtype vertex_colors_ROS: std_msgs.msg.ColorRGBA[]
        """
        mesh_ROS = Mesh()
        vertex_colors_ROS = []
        if vertex_colors is not None:
            for i in range(vertex_colors.shape[0]):
                v_color = ColorRGBA(vertex_colors[i, 0], vertex_colors[i, 1], vertex_colors[i, 2], 0)
                vertex_colors_ROS.append(v_color)
        for i in range(vertices.shape[0]):
            point = Point(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            mesh_ROS.vertices.append(point)
        for i in range(faces.shape[0]):
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices[0] = int(faces[i][0])
            mesh_triangle.vertex_indices[1] = int(faces[i][1])
            mesh_triangle.vertex_indices[2] = int(faces[i][2])
            mesh_ROS.triangles.append(mesh_triangle)
        return mesh_ROS, vertex_colors_ROS

    def from_ros_mesh(self, mesh_ROS, vertex_colors_ROS=None):
        """
        Converts a ROS mesh along with the corresponding colors of the vertices into a mesh
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
        vertex_colors = np.zeros([len(vertex_colors_ROS), 3])
        for i in range(len(mesh_ROS.vertices)):
            vertices[i] = np.array([mesh_ROS.vertices[i].x, mesh_ROS.vertices[i].y, mesh_ROS.vertices[i].z])
        for i in range(len(mesh_ROS.triangles)):
            faces[i] = np.array([int(mesh_ROS.triangles[i].vertex_indices[0]), int(mesh_ROS.triangles[i].vertex_indices[1]),
                                 int(mesh_ROS.triangles[i].vertex_indices[2])]).astype(int)
        if vertex_colors_ROS is not None:
            for i in range(len(vertex_colors_ROS)):
                vertex_colors[i] = np.array([vertex_colors_ROS[i].r, vertex_colors_ROS[i].g, vertex_colors_ROS[i].b])
        return vertices, faces, vertex_colors
