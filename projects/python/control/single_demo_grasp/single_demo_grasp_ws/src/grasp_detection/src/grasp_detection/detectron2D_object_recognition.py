#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import Image as CameraInfo
from opendr_bridge import ROSBridge
import os
import pathlib
from math import pi
from opendr.control.single_demo_grasp import SingleDemoGraspLearner, GraspDetection
from detectron2.data import MetadataCatalog, DatasetCatalog
from visualization_msgs.msg import Marker
from opendr.engine.data import Image

import tf 

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D, PoseStamped

import message_filters
import cv2


class Detectron2ObjectRecognitionNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", input_depth_image_topic="/usb_cam/image_raw", camera_info_topic="/camera/color/camera_info",
                 output_image_topic="/opendr/image_grasp_pose_annotated",
                 object_detection_topic="/opendr/object_detected",
                 grasp_detection_topic="/opendr/grasp_detected",
                 camera_tf_frame='/camera_color_frame', robot_tf_frame='panda_link0', ee_tf_frame='/panda_link8',
                 device="cuda"):
        """
        Creates a ROS Node for objects detection from RGBD
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param input_depth_image_topic: Topic from which we are reading the input depth image
        :type input_depth_image_topic: str
        :param gesture_annotations_topic: Topic to which we are publishing the predicted gesture class
        :type gesture_annotations_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        self.grasp_publisher = rospy.Publisher(grasp_detection_topic, ObjectHypothesisWithPose, queue_size=10)
        self.object_detection_publisher = rospy.Publisher(object_detection_topic, Detection2DArray, queue_size=10)
        self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)

        sub = rospy.Subscriber(input_image_topic, ROS_Image, self.callback)
        camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        self.bridge = ROSBridge()

        # Initialize the object detection        
        model_path = pathlib.Path("detectron5")
        print(model_path.exists())
        self.learner = SingleDemoGraspLearner(object_name='pendulum', data_directory=model_path, num_classes=4, device=device)
        if not model_path.exists():
            self.learner.download(path=model_path, object_name="pendulum")
        self.learner.load(model_path / "pendulum" / "output"/ "model_final.pth")
    
        self._classes = ["RockerArm","BoltHoles","Big_PushRodHoles","Small_PushRodHoles"]
        self._colors = []
        self._listener_tf = tf.TransformListener()
        self._camera_tf_frame = camera_tf_frame
        self._robot_tf_frame = robot_tf_frame
        self._ee_tf_frame = ee_tf_frame
        self._camera_focal = 550
        self._ctrX = 0
        self._ctrY = 0

    def camera_info_callback(self, msg):
        self.camera_focal = msg.K[0]
        self.ctrX = msg.K[2]
        self.ctrY = msg.K[5]

    def annotate(self, data, outputs):
        image = data.opencv()
        
        radius = 1
        thickness = 4
        
        if not self._colors:
            for i in range(len(self._classes)):
                self._colors.append(tuple(np.random.choice(range(255),size=3)))
   
        for detection in outputs:
            color = ( int(self._colors[detection.name][ 0 ]), int(self._colors[detection.name][ 1 ]), int(self._colors[detection.name][ 2 ])) 
            image = cv2.circle(image, (int(detection.data[0]),int(detection.data[1])), radius, color, thickness)
            clone = cv2.putText(image, "{}".format(int(detection.angle)), (int(detection.left+detection.width), int(detection.top+detection.height)), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            image = cv2.rectangle(image, (int(detection.left), int(detection.top)), (int(detection.left+detection.width), int(detection.top+detection.height)), color, 2)
        return image

    def to_ros_grasp_detections(self, object_detections):
        """
        Converts an OpenDR pose into a Detection2DArray msg that can carry the same information
        Each keypoint is represented as a bbox centered at the keypoint with zero width/height. The subject id is also
        embedded on each keypoint (stored in ObjectHypothesisWithPose).
        :param pose: OpenDR pose to be converted
        :type pose: engine.target.Pose
        :return: ROS message with the pose
        :rtype: vision_msgs.msg.Detection2DArray
        """
        ros_detection_array = Detection2DArray()
        for detection in object_detections:
            ros_detection = Detection2D()

            ros_detection.bbox = BoundingBox2D()
            keypoint = ObjectHypothesisWithPose()
            ros_detection.results.append(keypoint)

            ros_detection.bbox.center = Pose2D()
            ros_detection.bbox.center.x = detection.left + detection.width/2
            ros_detection.bbox.center.y = detection.top + detection.height/2

            ros_detection.bbox.size_x = detection.width
            ros_detection.bbox.size_y = detection.height

            ros_detection.results[0].id = detection.name
            if detection.confidence:
                ros_detection.results[0].score = detection.confidence
            ros_detection.results[0].pose.pose.position.x = detection.data[0]
            ros_detection.results[0].pose.pose.position.y = detection.data[1]
            ros_detection_array.detections.append(ros_detection)
        return ros_detection_array

    def to_ros_grasp_pose(self, obj_id, pose_stamped):
        grasp_pose = ObjectHypothesisWithPose()
        grasp_pose.id = obj_id
        grasp_pose.pose.pose = pose_stamped.pose
        return grasp_pose

    def convert_detection_pose(self, object_detection_2D, z_to_surface, rot, marker_id):
        to_world_scale = z_to_surface/ self._camera_focal

        x_dist = (object_detection_2D.data[0]-630) * to_world_scale
        y_dist = (902-object_detection_2D.data[1]) * to_world_scale

        my_point = PoseStamped()
        my_point.header.frame_id = self._camera_tf_frame
        my_point.header.stamp = rospy.Time(0)
        '''
        my_point.pose.position.x = -y_dist
        my_point.pose.position.y = x_dist
        my_point.pose.position.z = -0.15
        '''
        my_point.pose.position.x = 0
        my_point.pose.position.y = x_dist
        my_point.pose.position.z = y_dist

        theta = object_detection_2D.angle / 180 * pi
        #theta = 0
        ps = self._listener_tf.lookupTransform(self._camera_tf_frame, self._ee_tf_frame, rospy.Time(0))
        quat_rotcmd = tf.transformations.quaternion_from_euler(theta, 0, 0)
        quat = tf.transformations.quaternion_multiply(quat_rotcmd, ps[1])

        my_point.pose.orientation.x = quat[0]
        my_point.pose.orientation.y = quat[1]
        my_point.pose.orientation.z = quat[2]
        my_point.pose.orientation.w = quat[3]
        
        ps=self._listener_tf.transformPose(self._robot_tf_frame,my_point)

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
        marker.pose.position.x = ps.pose.position.x
        marker.pose.position.y = ps.pose.position.y
        marker.pose.position.z = ps.pose.position.z
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        self.marker_pub.publish(marker)
        return ps


    def callback(self, image_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param image_data: input image message
        :type image_data: sensor_msgs.msg.Image
        :param depth_data: input depth image message
        :type depth_data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image and preprocess
        image = self.bridge.from_ros_image(image_data)
        # Run gesture recognition
        object_detections = self.learner.infer(image)

        #  Publish results
        annotated_image = self.annotate(image, object_detections)
        image_message = self.bridge.to_ros_image(Image(annotated_image), encoding='bgr8')
        self.image_publisher.publish(image_message)

        ros_object_detections = self.to_ros_grasp_detections(object_detections)
        self.object_detection_publisher.publish(ros_object_detections)
        # If a robot is connected, convert poses
        try:
            (trans, rot) = self._listener_tf.lookupTransform(self._robot_tf_frame, self._camera_tf_frame, rospy.Time(0))
            print("Camera transform")
            print(trans)
            print(rot)
        except Exception as e:
            print("No robot found. Skipping pose conversion.")
        else:
            i = 0
            for detection in object_detections:
                ros_pose = self.convert_detection_pose(detection, trans[2]-0.10, rot, i)
                ros_grasp_detection = self.to_ros_grasp_pose(detection.name, ros_pose)
                self.grasp_publisher.publish(ros_grasp_detection)
                i+=1



    def preprocess(self, image, depth_img):
        '''
        Preprocess image, depth_image and concatenate them
        :param image_data: input image
        :type image_data: engine.data.Image
        :param depth_data: input depth image
        :type depth_data: engine.data.Image
        '''
        image = image.convert(format='channels_last') / (2**8 - 1)
        depth_img = depth_img.convert(format='channels_last') / (2**16 - 1)

        # resize the images to 224x224
        image = cv2.resize(image, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))

        # concatenate and standardize
        img = np.concatenate([image, np.expand_dims(depth_img, axis=-1)], axis=-1)
        img = (img - self.mean) / self.std
        img = Image(img, dtype=np.float32)
        return img

if __name__ == '__main__':

    rospy.init_node('opendr_object_detection', anonymous=True)

    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    # default topics for intel realsense - https://github.com/IntelRealSense/realsense-ros
    depth_topic = "/camera/depth/image_rect_raw"
    image_topic = "/camera/color/image_raw"  

    object_node = Detectron2ObjectRecognitionNode(input_image_topic=image_topic, input_depth_image_topic=depth_topic, device=device)

    rospy.loginfo("Detectron2 object detection node started!")
    rospy.spin()

