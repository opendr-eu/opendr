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

import numpy as np
from opendr.engine.data import Image, PointCloud, Timeseries
from opendr.engine.target import (
    Pose, BoundingBox, BoundingBoxList, Category,
    BoundingBox3D, BoundingBox3DList, TrackingAnnotation
)
from cv_bridge import CvBridge
from std_msgs.msg import String, ColorRGBA, Header
from sensor_msgs.msg import Image as ImageMsg, PointCloud as PointCloudMsg, ChannelFloat32 as ChannelFloat32Msg
from vision_msgs.msg import (
    Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose,
    Detection3D, Detection3DArray, BoundingBox3D as BoundingBox3DMsg,
    Classification2D, ObjectHypothesis
)
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import (
    Pose2D, Point32 as Point32Msg,
    Quaternion as QuaternionMsg, Pose as Pose3D,
    Point
)
from opendr_interface.msg import OpenDRPose2D, OpenDRPose2DKeypoint, OpenDRPose3D, OpenDRPose3DKeypoint


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
        :rtype: opendr_interface.msg.OpenDRPose2D
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
        :type ros_pose: opendr_interface.msg.OpenDRPose2D
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

    def from_ros_single_tracking_annotation(self, ros_detection_box):
        """
        Converts a pair of ROS messages with bounding boxes and tracking ids into an OpenDR TrackingAnnotationList
        :param ros_detection_box: The boxes to be converted.
        :type ros_detection_box: vision_msgs.msg.Detection2D
        :return: An OpenDR TrackingAnnotationList
        :rtype: engine.target.TrackingAnnotationList
        """
        width = ros_detection_box.bbox.size_x
        height = ros_detection_box.bbox.size_y
        left = ros_detection_box.bbox.center.x - width / 2.
        top = ros_detection_box.bbox.center.y - height / 2.
        id = 0
        bbox = TrackingAnnotation(
            name=id,
            left=left,
            top=top,
            width=width,
            height=height,
            id=0,
            frame=-1
        )
        return bbox

    def to_ros_single_tracking_annotation(self, tracking_annotation):
        """
        Converts a pair of ROS messages with bounding boxes and tracking ids into an OpenDR TrackingAnnotationList
        :param tracking_annotation: The box to be converted.
        :type tracking_annotation: engine.target.TrackingAnnotation
        :return: A ROS vision_msgs.msg.Detection2D
        :rtype: vision_msgs.msg.Detection2D
        """
        ros_box = Detection2D()
        ros_box.bbox = BoundingBox2D()
        ros_box.results.append(ObjectHypothesisWithPose())
        ros_box.bbox.center = Pose2D()
        ros_box.bbox.center.x = tracking_annotation.left + tracking_annotation.width / 2.0
        ros_box.bbox.center.y = tracking_annotation.top + tracking_annotation.height / 2.0
        ros_box.bbox.size_x = float(tracking_annotation.width)
        ros_box.bbox.size_y = float(tracking_annotation.height)
        ros_box.results[0].id = str(tracking_annotation.name)
        ros_box.results[0].score = float(-1)
        return ros_box

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
        :param point_cloud: ROS PointCloud to be converted
        :type point_cloud: sensor_msgs.msg.PointCloud
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
        :param point_cloud: OpenDR PointCloud
        :type point_cloud: engine.data.PointCloud
        :param time_stamp: Time stamp
        :type time_stamp: ROS Time
        :return: ROS PointCloud
        :rtype: sensor_msgs.msg.PointCloud
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

    def from_ros_mesh(self, mesh_ROS):
        """
        Converts a ROS mesh into arrays of vertices and faces of a mesh
        :param mesh_ROS: the ROS mesh to be converted
        :type mesh_ROS: shape_msgs.msg.Mesh
        :return vertices: Numpy array Nx3 representing vertices of the 3D model respectively
        :rtype vertices: np.array
        :return faces: Numpy array Nx3 representing the IDs of the vertices of each face of the 3D model
        :rtype faces: numpy array (Nx3)
        """
        vertices = np.zeros([len(mesh_ROS.vertices), 3])
        faces = np.zeros([len(mesh_ROS.triangles), 3]).astype(int)
        for i in range(len(mesh_ROS.vertices)):
            vertices[i] = np.array([mesh_ROS.vertices[i].x, mesh_ROS.vertices[i].y, mesh_ROS.vertices[i].z])
        for i in range(len(mesh_ROS.triangles)):
            faces[i] = np.array([int(mesh_ROS.triangles[i].vertex_indices[0]), int(mesh_ROS.triangles[i].vertex_indices[1]),
                                 int(mesh_ROS.triangles[i].vertex_indices[2])]).astype(int)
        return vertices, faces

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
            point = Point()
            point.x = vertices[i, 0]
            point.y = vertices[i, 1]
            point.z = vertices[i, 2]
            mesh_ROS.vertices.append(point)
        for i in range(faces.shape[0]):
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices[0] = int(faces[i][0])
            mesh_triangle.vertex_indices[1] = int(faces[i][1])
            mesh_triangle.vertex_indices[2] = int(faces[i][2])
            mesh_ROS.triangles.append(mesh_triangle)
        return mesh_ROS

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

    def to_ros_colors(self, colors):
        """
        Converts an array of vertex_colors to a list of ROS colors
        :param colors: a numpy array of RGB colors
        :type colors: numpy array (Nx3)
        :return ros_colors: a list of the colors of the vertices
        :rtype ros_colors: std_msgs.msg.ColorRGBA[]
        """
        ros_colors = []
        for i in range(colors.shape[0]):
            color = ColorRGBA()
            color.r = colors[i, 0]
            color.g = colors[i, 1]
            color.b = colors[i, 2]
            color.a = 0.0
            ros_colors.append(color)
        return ros_colors

    def from_ros_pose_3D(self, ros_pose):
        """
        Converts a ROS message with pose payload into an OpenDR pose
        :param ros_pose: the pose to be converted (represented as opendr_interface.msg.OpenDRPose3D)
        :type ros_pose: opendr_interface.msg.OpenDRPose3D
        :return: an OpenDR pose
        :rtype: engine.target.Pose
        """
        keypoints = ros_pose.keypoint_list
        data = []
        for i, keypoint in enumerate(keypoints):
            data.append([keypoint.x, keypoint.y, keypoint.z])
        pose = Pose(data, 1.0)
        pose.id = 0
        return pose

    def to_ros_pose_3D(self, pose):
        """
        Converts an OpenDR pose into a OpenDRPose3D msg that can carry the same information
        Each keypoint is represented as an OpenDRPose3DKeypoint with x, y, z coordinates.
        :param pose: OpenDR pose to be converted
        :type pose: engine.target.Pose
        :return: ROS message with the pose
        :rtype: opendr_interface.msg.OpenDRPose3D
        """
        data = pose.data
        ros_pose = OpenDRPose3D()
        ros_pose.pose_id = 0
        if pose.id is not None:
            ros_pose.pose_id = int(pose.id)
        ros_pose.conf = 1.0
        for i in range(len(data)):
            keypoint = OpenDRPose3DKeypoint()
            keypoint.kpt_name = ''
            keypoint.x = float(data[i][0])
            keypoint.y = float(data[i][1])
            keypoint.z = float(data[i][2])
            ros_pose.keypoint_list.append(keypoint)
        return ros_pose

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
        result.score = float(category.confidence)
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

    def from_ros_image_to_depth(self, message, encoding='mono16'):
        """
        Converts a ROS2 image message into an OpenDR grayscale depth image
        :param message: ROS2 image to be converted
        :type message: sensor_msgs.msg.Image
        :param encoding: encoding to be used for the conversion
        :type encoding: str
        :return: OpenDR image
        :rtype: engine.data.Image
        """
        cv_image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        cv_image = np.expand_dims(cv_image, axis=-1)
        image = Image(np.asarray(cv_image, dtype=np.uint8))
        return image

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
