#!/usr/bin/env python3
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

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
import message_filters
import pathlib
from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, VisionInfo
from opendr_ros2_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.engine.target import BoundingBoxList
from opendr.perception.object_detection_2d import Detectron2Learner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class Detectron2GraspDetectionNode(Node):

    def __init__(self, camera_tf_frame, robot_tf_frame, ee_tf_frame,
                 input_image_topic="/usb_cam/image_raw",
                 input_depth_image_topic="/usb_cam/image_raw",
                 camera_info_topic="/camera/color/camera_info",
                 output_image_topic="/opendr/image_grasp_pose_annotated",
                 output_mask_topic="/opendr/image_mask_annotated",
                 object_detection_topic="/opendr/object_detected",
                 grasp_detection_topic="/opendr/grasp_detected",
                 device="cuda", only_visualize=True, model="detectron"):
        """
        Creates a ROS Node for object detection with YOLOV3
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no object detection message
        is published)
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: backbone network
        :type backbone: str
        """
        super().__init__('detectron2_grasp_detection_node')

        self.declare_parameters(
            namespace='opendr',
            parameters=[
                ('object_catagories', None)
            ])

        model_path = pathlib.Path("/home/opendr/Gaurang/engineAssembly/new_dataset/output_level2")
        # model_path = pathlib.Path("/home/opendr/augmentation/output")
        if model == "detectron":
            self.learner = Detectron2Learner(device=device)
            if not model_path.exists():
                self.learner.download(path=model_path)
            self.learner.load(model_path / "model_final.pth")

        self.bridge = ROS2Bridge()

        self.object_publisher = self.create_publisher(Detection2DArray, object_detection_topic, 1)
        self.grasp_publisher = self.create_publisher(ObjectHypothesisWithPose, grasp_detection_topic, 10)
        self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 1)
        self.mask_publisher = self.create_publisher(ROS_Image, output_mask_topic, 1)
        self.obj_cat_pub = self.create_publisher(VisionInfo, "/opendr/object_catagories", 1)
        # self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 2)
        image_sub = message_filters.Subscriber(self, ROS_Image, input_image_topic, qos_profile=1)
        depth_sub = message_filters.Subscriber(self, ROS_Image, input_depth_image_topic, qos_profile=1)
        # synchronize image and depth data topics
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.callback)

        self.only_visualize = only_visualize
        self._colors = []
        # self._listener_tf = tf.TransformListener()
        self._camera_tf_frame = camera_tf_frame
        self._robot_tf_frame = robot_tf_frame
        self._ee_tf_frame = ee_tf_frame

        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.obj_cat_callback)

        self.get_logger().info("Detectron2 object detection node started!")

    def obj_cat_callback(self):
        self.obj_cat_pub.publish(VisionInfo(method="object categories",
                                            database_location="/opendr/object_categories"))

    def camera_info_callback(self, msg):
        self._camera_focal = msg.K[0]
        self._refX = msg.K[2]
        self._refY = msg.K[5]

    '''
    def create_rviz_marker(self, ros_pose, marker_id):
        marker = Marker()
        marker.header.frame_id = self._robot_tf_frame
        marker.header.stamp = rospy.Time.now()
        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = marker_id
        # Set the scale of the marker
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = ros_pose.pose.position.x
        marker.pose.position.y = ros_pose.pose.position.y
        marker.pose.position.z = ros_pose.pose.position.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        self.marker_pub.publish(marker)

    def compute_orientation(self, seg_mask):
        # M = cv2.moments(seg_mask)

        # calculate x,y coordinate of center
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])

        y, x = np.nonzero(seg_mask)
        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        # theta = np.arctan((x_v1)/(y_v
        theta = math.atan2(y_v1, x_v1)
        # theta = math.degrees(theta)
        return theta

    def compute_depth(self, depth_image, seg_mask):
        seg_mask = seg_mask.astype('bool')
        object_depth_map = depth_image[seg_mask].tolist()
        depth_values = [x for x in object_depth_map if x != 0]
        unique, frequency = np.unique(depth_values, return_counts=True)
        depth = unique[np.where(frequency == max(frequency))]
        real_z = depth[0]/1000
        return real_z

    def get_center_bbox(self, bbox):
        ctr_x = bbox.left+bbox.width/2
        ctr_y = bbox.top+bbox.height/2
        return (ctr_x, ctr_y)

    def convert_detection_pose(self, bbox, z_to_surface, rot):
        print(z_to_surface)
        to_world_scale = z_to_surface / self._camera_focal

        ctr_x, ctr_y = self.get_center_bbox(bbox)
        x_dist = (ctr_x - self._refX) * to_world_scale
        y_dist = (self._refY - ctr_y) * to_world_scale

        my_point = PoseStamped()
        my_point.header.frame_id = self._camera_tf_frame
        my_point.header.stamp = rospy.Time(0)

        my_point.pose.position.x = -x_dist
        my_point.pose.position.y = y_dist
        my_point.pose.position.z = 0

        quat_rotcmd = tf.transformations.quaternion_from_euler(0, 0, 0)
        my_point.pose.orientation.x = quat_rotcmd[0]
        my_point.pose.orientation.y = quat_rotcmd[1]
        my_point.pose.orientation.z = quat_rotcmd[2]
        my_point.pose.orientation.w = quat_rotcmd[3]

        ps=self._listener_tf.transformPose(self._robot_tf_frame,my_point)
        return ps
    '''

    def draw_seg_masks(self, image, masks):
        masked_image = np.array([])
        if masks:
            masked_image = np.full_like(masks[0], 0)
            for mask in masks:
                mask = np.asarray(mask)
                masked_image += mask
        return masked_image

    def callback(self,  rgb_data, depth_data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(rgb_data)
        depth_data.encoding = "mono16"
        depth_image = self.bridge._cv_bridge.imgmsg_to_cv2(depth_data,
                                                           desired_encoding='mono16')
        # Run object detection
        object_detections = self.learner.infer(image)
        boxes = BoundingBoxList([box for kp, box, seg_mask in object_detections])
        seg_masks = [seg_mask for kp, box, seg_mask in object_detections]

        # Get an OpenCV image back
        image = image.opencv()

        # Publish detections in ROS message
        ros_boxes = self.bridge.to_ros_bounding_box_list(boxes)  # Convert to ROS bounding_box_list
        if self.object_publisher is not None:
            self.object_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            # Annotate image with object detection boxes
            image = draw_bounding_boxes(image, boxes)
            masks = self.draw_seg_masks(image, seg_masks)
            # Convert the annotated OpenDR image to ROS image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image),
                                         encoding='bgr8'))
            if not masks.size == 0:
                masked_image = np.zeros_like(image)
                masked_image[:, :, 0] = masks[:]
                masked_image[:, :, 1] = masks[:]
                masked_image[:, :, 2] = masks[:]
                self.mask_publisher.publish(self.bridge.to_ros_image(Image(masked_image)))
            else:
                self.mask_publisher.publish(self.bridge.to_ros_image(Image(image),
                                            encoding='bgr8'))
        '''
        if not self.only_visualize:
            i = 0
            for bbox, mask in zip(boxes, seg_masks):
                theta = self.compute_orientation(mask)
                mask = np.asarray(mask)
                depth = self.compute_depth(depth_image, mask)
                ros_pose = self.convert_detection_pose(bbox, depth, theta)
                self.create_rviz_marker(ros_pose, i)
                self.grasp_publisher.publish(ros_pose)
                i += 1
        '''


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_topic', default="/camera/color/image_raw", type=str,
                        help='ROS topic containing RGB information')
    parser.add_argument('--depth_topic', default="/camera/aligned_depth_to_color/image_raw", type=str,
                        help='ROS topic containing depth information')
    parser.add_argument('--camera_info_topic', default="/camera/color/camera_info", type=str,
                        help='ROS topic containing meta information')
    parser.add_argument('--camera_tf_frame', default="/camera_color_optical_frame", type=str,
                        help='Tf frame in which objects are detected')
    parser.add_argument('--robot_tf_frame', default="panda_link0", type=str,
                        help='Tf frame of reference for commands sent to the robot')
    parser.add_argument('--ee_tf_frame', default="/panda_link8", type=str,
                        help='Tf frame of reference for the robot end effector')
    parser.add_argument('--only_visualize', default=True, type=bool,
                        help='''True if running the detection for visualizing purposes only,
                        False to convert detections to a robot coordinates frame''')

    args = parser.parse_args()

    device = "cuda"
    grasp_detection_node = Detectron2GraspDetectionNode(camera_tf_frame=args.camera_tf_frame,
                                                        robot_tf_frame=args.robot_tf_frame,
                                                        ee_tf_frame=args.ee_tf_frame,
                                                        input_image_topic=args.image_topic,
                                                        input_depth_image_topic=args.depth_topic,
                                                        device=device,
                                                        only_visualize=args.only_visualize,
                                                        camera_info_topic=args.camera_info_topic)

    rclpy.spin(grasp_detection_node)
    grasp_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
