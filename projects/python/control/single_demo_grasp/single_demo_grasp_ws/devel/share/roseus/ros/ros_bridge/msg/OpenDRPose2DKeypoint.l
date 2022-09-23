;; Auto-generated. Do not edit!


(when (boundp 'ros_bridge::OpenDRPose2DKeypoint)
  (if (not (find-package "ROS_BRIDGE"))
    (make-package "ROS_BRIDGE"))
  (shadow 'OpenDRPose2DKeypoint (find-package "ROS_BRIDGE")))
(unless (find-package "ROS_BRIDGE::OPENDRPOSE2DKEYPOINT")
  (make-package "ROS_BRIDGE::OPENDRPOSE2DKEYPOINT"))

(in-package "ROS")
;;//! \htmlinclude OpenDRPose2DKeypoint.msg.html


(defclass ros_bridge::OpenDRPose2DKeypoint
  :super ros::object
  :slots (_kpt_name _x _y ))

(defmethod ros_bridge::OpenDRPose2DKeypoint
  (:init
   (&key
    ((:kpt_name __kpt_name) "")
    ((:x __x) 0)
    ((:y __y) 0)
    )
   (send-super :init)
   (setq _kpt_name (string __kpt_name))
   (setq _x (round __x))
   (setq _y (round __y))
   self)
  (:kpt_name
   (&optional __kpt_name)
   (if __kpt_name (setq _kpt_name __kpt_name)) _kpt_name)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:serialization-length
   ()
   (+
    ;; string _kpt_name
    4 (length _kpt_name)
    ;; int32 _x
    4
    ;; int32 _y
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _kpt_name
       (write-long (length _kpt_name) s) (princ _kpt_name s)
     ;; int32 _x
       (write-long _x s)
     ;; int32 _y
       (write-long _y s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _kpt_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _kpt_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; int32 _x
     (setq _x (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _y
     (setq _y (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get ros_bridge::OpenDRPose2DKeypoint :md5sum-) "6ce75e74f73663ed82a3235764bc7edf")
(setf (get ros_bridge::OpenDRPose2DKeypoint :datatype-) "ros_bridge/OpenDRPose2DKeypoint")
(setf (get ros_bridge::OpenDRPose2DKeypoint :definition-)
      "# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This message contains all relevant information for an OpenDR human pose 2D keypoint

# The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes
string kpt_name

# x and y pixel position on the input image, (0, 0) is top-left corner of image
int32 x
int32 y

")



(provide :ros_bridge/OpenDRPose2DKeypoint "6ce75e74f73663ed82a3235764bc7edf")


