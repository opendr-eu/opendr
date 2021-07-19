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

import sys
import copy
import cv2
import rospy
import roslib
import numpy as np
import time
from std_msgs.msg import String, Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from single_demo_inference import *
CUDA_LAUNCH_BLOCKING=1


class SingleDemoGraspCameraStream(object):

   """SingleDemoGraspCameraStream initialization"""
   def __init__(self, path_to_dt_model, thresh):

        self.object_locator = SingleDemoInference(path_to_dt_model, thresh)
        self.rgb_image = cv2.imread("samples/0.jpg")
        self.command_publisher = rospy.Publisher('/commands', Float32MultiArray, queue_size=1)
        self.detection_request_sub = rospy.Subscriber("/request_detection", Int16,
                                                            self.request_callback)
        self.image_sub=  rospy.Subscriber("/camera/color/raw",Image, self.image_callback)


   def image_callback(self, data):
       self.rgb_image = CvBridge().imgmsg_to_cv2(data, desired_encoding="rgb8")

   def request_callback(self, data):

        print ("new request:")
        print (data.data)
        image_analyze(data.data)


   def image_analyze(self, msg_id):


        analyze_img = self.rgb_image.copy()
        flag, bbx, pred_angle, pred_kps_center = self.object_locator.predict(analyze_img)
        bbx = np.asarray(bbx)
        bbx = bbx.astype(int)
        msg = Float32MultiArray()

        if (flag > 0):

            print (bbx)
            ctr_X = int((bbx[0]+bbx[2])/2)
            ctr_Y = int((bbx[1]+bbx[3])/2)
            angle = pred_angle
            ref_x = 640/2
            ref_y = 480/2

            # distance to the center of bounding box representing the center of object
            dist = [ctr_X - ref_x, ref_y - ctr_Y]
            # distance center of keypoints representing the grasp location of the object
            dist_kps_ctr = [pred_kps_center[0] - ref_x, ref_y - pred_kps_center[1]]
            msg.data = [msg_id, dist[0], dist[1], angle, dist_kps_ctr[0],
                                                    dist_kps_ctr[1]]
            self.command_publisher.publish(msg)

        else:
            # 1e10 as a big large enough number out of range. reciever use this number
            # to check whether a detection is available or not
            msg.data=[msg_id,1e10,1e10,1e10,1e10]
            self.command_publisher.publish(msg)



if __name__ == '__main__':

    rospy.init_node('grasp_server', anonymous=True)
    camera_streamer = SingleDemoGraspCameraStream(sys.argv[1], 0.8)
    rospy.spin()
    input()
    sys.exit()
