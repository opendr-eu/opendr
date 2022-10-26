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
from opendr.perception.object_detection_2d import draw_bounding_boxes

import tf 

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D, PoseStamped

import message_filters
import cv2


def get_kps_center(input_kps) :
    input_kps_sorted = input_kps[input_kps[:, 2].argsort()[::-1]]
    kps_x_list = input_kps[:,0]
    kps_y_list = input_kps[:,1]
    kps_x_list = kps_x_list[::-1]
    kps_y_list = kps_y_list[::-1]
    d = {'X' : kps_x_list,
        'Y' : kps_y_list}
    df = pd.DataFrame(data = d)

    x = np.mean(df["X"][0:2])
    y= np.mean(df["Y"][0:2])
    return [x,y]

def get_angle(input_kps, mode):
    input_kps_sorted = input_kps[input_kps[:, 2].argsort()[::-1]]
    # kps_x_list = input_kps[0][:,0]
    # kps_y_list = input_kps[0][:,1]
    kps_x_list = input_kps_sorted[:,0]
    kps_y_list = input_kps_sorted[:,1]
    kps_x_list = kps_x_list[::-1]
    kps_y_list = kps_y_list[::-1]

    d = {'X' : kps_x_list,
        'Y' : kps_y_list}

    df = pd.DataFrame(data = d)


    # move the origin
    x = (df["X"] - df["X"][0])
    y = (df["Y"] - df["Y"][0])

    #x = x[0:2]
    #y = y[0:2]
    if mode == 1:
        list_xy = (np.arctan2(y,x)*180/math.pi).astype(int)
        occurence_count = Counter(list_xy)
        return occurence_count.most_common(1)[0][0]
    else:
        x = np.mean(x)
        y= np.mean(y)
        return np.arctan2(y,x)*180/math.pi

def correct_orientation_ref(angle):

    if angle <= 90:
        angle += 90
    if angle > 90:
        angle += -270

    return angle


class Detectron2ObjectRecognitionNode:

    def __init__(self, camera_tf_frame, robot_tf_frame, ee_tf_frame,
                 input_image_topic="/usb_cam/image_raw", input_depth_image_topic="/usb_cam/image_raw", 
                 camera_info_topic="/camera/color/camera_info",
                 output_image_topic="/opendr/image_grasp_pose_annotated",
                 object_detection_topic="/opendr/object_detected",
                 grasp_detection_topic="/opendr/grasp_detected",
                 device="cuda", model="detectron"):
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

        self.object_publisher = rospy.Publisher(object_detection_topic, Detection2DArray, queue_size=1)
        self.grasp_publisher = rospy.Publisher(grasp_detection_topic, ObjectHypothesisWithPose, queue_size=10)
        self.image_publisher = rospy.Publisher(output_image_topic, ROS_Image, queue_size=1)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)

        sub = rospy.Subscriber(input_image_topic, ROS_Image, self.callback)
        camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        self.bridge = ROSBridge()

        # Initialize the object detection        
        model_path = pathlib.Path("detectron")
        
        if model == "detectron":
            self.learner = SingleDemoGraspLearner(data_directory=model_path, device=device)
            if not model_path.exists():
                self.learner.download(path=model_path)
            self.learner.load(model_path / "output"/ "model_final.pth")
    
        self._classes = ["RockerArm","BoltHoles","Big_PushRodHoles","Small_PushRodHoles", "Engine", "Bolt","PushRod","RockerArmObject"]
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

    def create_rviz_marker(self, ros_pose):
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

    def convert_detection_pose(self, object_detection_2D, z_to_surface, rot, marker_id):
        to_world_scale = z_to_surface/ self._camera_focal

        x_dist = (object_detection_2D.data[0]-self.ctrX) * to_world_scale
        y_dist = (self.ctrY-object_detection_2D.data[1]) * to_world_scale

        '''
        ctr_X = int((bbx[0] + bbx[2]) / 2)
        ctr_Y = int((bbx[1] + bbx[3]) / 2)
        angle = pred_angle
        ref_x = 640 / 2
        ref_y = 480 / 2
        # distance to the center of bounding box representing the center of object
        dist = [ctr_X - ref_x, ref_y - ctr_Y]
        # distance center of keypoints representing the grasp location of the object
        dist_kps_ctr = [pred_kps_center[0] - ref_x, ref_y - pred_kps_center[1]]
        '''    
        
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
        
        # Run object detection
        object_detections = self.learner.infer(image)
        boxes = BoundingBoxList([box for kp,box in detections])
        
        # Get an OpenCV image back
        image = image.opencv()

        # Publish detections in ROS message
        ros_boxes = self.bridge.to_ros_bounding_box_list(boxes)  # Convert to ROS bounding_box_list
        if self.object_publisher is not None:
            self.object_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            # Annotate image with object detection boxes
            image = draw_bounding_boxes(image, boxes)
            # Convert the annotated OpenDR image to ROS image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


        # If a robot is connected, convert poses
        try:
            (trans, rot) = self._listener_tf.lookupTransform(self._robot_tf_frame, self._camera_tf_frame, rospy.Time(0))
        except Exception as e:
            print("No robot found. Skipping pose conversion.")
        else:
            i = 0
            for detection in object_detections:
                ros_pose = self.convert_detection_pose(detection, trans[2]-0.10, rot, i)
                self.create_rviz_marker(ros_pose)
                ros_grasp_detection = self.bridg#e.to_ros_grasp_pose(detection.name, ros_pose)
                self.grasp_publisher.publish(ros_grasp_detection)
                i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_tf_frame', type=str, help='Tf frame in which objects are detected')
    parser.add_argument('--robot_tf_frame', type=str, help='Tf frame of reference for commands sent to the robot')
    parser.add_argument('--ee_tf_frame', type=str, help='Tf frame of reference for the robot end effector')
    parser.add_argument('--depth_topic', type=str, help='ROS topic containing depth information')
    parser.add_argument('--image_topic', type=str, help='ROS topic containing RGB information')
    args = parser.parse_args()

    rospy.init_node('opendr_object_detection', anonymous=True)
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'
    # default topics for intel realsense - https://github.com/IntelRealSense/realsense-ros
    depth_topic = "/camera/depth/image_rect_raw"
    image_topic = "/camera/color/image_raw"  
    object_node = Detectron2ObjectRecognitionNode(camera_tf_frame='/camera_color_frame', robot_tf_frame='panda_link0', ee_tf_frame='/panda_link8', input_image_topic=image_topic, input_depth_image_topic=depth_topic, device=device)
    rospy.loginfo("Detectron2 object detection node started!")
    rospy.spin()
