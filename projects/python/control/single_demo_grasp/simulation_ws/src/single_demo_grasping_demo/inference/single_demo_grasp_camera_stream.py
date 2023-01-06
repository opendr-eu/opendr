#!/usr/bin/env python

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

import os
import sys
import rospy
import numpy as np
from std_msgs.msg import Int16
from sensor_msgs.msg import Image as ROS_Image
from std_msgs.msg import Float32MultiArray
from single_demo_inference import SingleDemoInference
from opendr_bridge import ROSBridge


class SingleDemoGraspCameraStream(object):

    def __init__(self, path_to_dt_model, thresh):
        """SingleDemoGraspCameraStream initialization"""
        self.object_locator = SingleDemoInference(path_to_dt_model, thresh)
        self.rgb_image = None
        self.command_publisher = rospy.Publisher('/commands', Float32MultiArray, queue_size=1)
        self.detection_request_sub = rospy.Subscriber("/request_detection", Int16, self.request_callback)
        self.image_sub = rospy.Subscriber("/camera/color/raw", ROS_Image, self.image_callback)
        self.bridge = ROSBridge()

    def image_callback(self, data):
        self.rgb_image = self.bridge.from_ros_image(data, encoding='rgb8')

    def request_callback(self, data):
        print("new request:")
        print(data.data)
        self.image_analyze(data.data)

    def image_analyze(self, msg_id):
        analyze_img = self.rgb_image.opencv()
        flag, bbx, pred_angle, pred_kps_center = self.object_locator.predict(analyze_img)
        bbx = np.asarray(bbx)
        bbx = bbx.astype(int)
        msg = Float32MultiArray()

        if (flag > 0):
            print(bbx)
            ctr_X = int((bbx[0] + bbx[2]) / 2)
            ctr_Y = int((bbx[1] + bbx[3]) / 2)
            angle = pred_angle
            ref_x = 640 / 2
            ref_y = 480 / 2

            # distance to the center of bounding box representing the center of object
            dist = [ctr_X - ref_x, ref_y - ctr_Y]
            # distance center of keypoints representing the grasp location of the object
            dist_kps_ctr = [pred_kps_center[0] - ref_x, ref_y - pred_kps_center[1]]
            msg.data = [msg_id, dist[0], dist[1], angle, dist_kps_ctr[0], dist_kps_ctr[1]]
            self.command_publisher.publish(msg)

        else:
            # 1e10 as a big large enough number out of range. reciever use this number
            # to check whether a detection is available or not
            msg.data = [msg_id, 1e10, 1e10, 1e10, 1e10]
            self.command_publisher.publish(msg)


if __name__ == '__main__':

    dir_temp = os.path.join("./", "sdg_temp")
    rospy.init_node('grasp_server', anonymous=True)
    camera_streamer = SingleDemoGraspCameraStream(os.path.join(dir_temp, "pendulum", "output", "model_final.pth"), 0.8)
    rospy.spin()
    input()
    sys.exit()
