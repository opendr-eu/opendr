;; Auto-generated. Do not edit!


(when (boundp 'ros_bridge::OpenDRPose2D)
  (if (not (find-package "ROS_BRIDGE"))
    (make-package "ROS_BRIDGE"))
  (shadow 'OpenDRPose2D (find-package "ROS_BRIDGE")))
(unless (find-package "ROS_BRIDGE::OPENDRPOSE2D")
  (make-package "ROS_BRIDGE::OPENDRPOSE2D"))

(in-package "ROS")
;;//! \htmlinclude OpenDRPose2D.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass ros_bridge::OpenDRPose2D
  :super ros::object
  :slots (_header _pose_id _conf _keypoint_list ))

(defmethod ros_bridge::OpenDRPose2D
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:pose_id __pose_id) 0)
    ((:conf __conf) 0.0)
    ((:keypoint_list __keypoint_list) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _pose_id (round __pose_id))
   (setq _conf (float __conf))
   (setq _keypoint_list __keypoint_list)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:pose_id
   (&optional __pose_id)
   (if __pose_id (setq _pose_id __pose_id)) _pose_id)
  (:conf
   (&optional __conf)
   (if __conf (setq _conf __conf)) _conf)
  (:keypoint_list
   (&rest __keypoint_list)
   (if (keywordp (car __keypoint_list))
       (send* _keypoint_list __keypoint_list)
     (progn
       (if __keypoint_list (setq _keypoint_list (car __keypoint_list)))
       _keypoint_list)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int32 _pose_id
    4
    ;; float32 _conf
    4
    ;; ros_bridge/OpenDRPose2DKeypoint[] _keypoint_list
    (apply #'+ (send-all _keypoint_list :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int32 _pose_id
       (write-long _pose_id s)
     ;; float32 _conf
       (sys::poke _conf (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; ros_bridge/OpenDRPose2DKeypoint[] _keypoint_list
     (write-long (length _keypoint_list) s)
     (dolist (elem _keypoint_list)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int32 _pose_id
     (setq _pose_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _conf
     (setq _conf (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; ros_bridge/OpenDRPose2DKeypoint[] _keypoint_list
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _keypoint_list (let (r) (dotimes (i n) (push (instance ros_bridge::OpenDRPose2DKeypoint :init) r)) r))
     (dolist (elem- _keypoint_list)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get ros_bridge::OpenDRPose2D :md5sum-) "88f7162365f7e82118b9fd3fc8f9ae3b")
(setf (get ros_bridge::OpenDRPose2D :datatype-) "ros_bridge/OpenDRPose2D")
(setf (get ros_bridge::OpenDRPose2D :definition-)
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

# This message represents a full OpenDR human pose 2D as a list of keypoints

Header header

# The id of the pose
int32 pose_id

# The pose detection confidence of the model
float32 conf

# A list of a human 2D pose keypoints
OpenDRPose2DKeypoint[] keypoint_list

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: ros_bridge/OpenDRPose2DKeypoint
# Copyright 2020-2022 OpenDR European Project
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



(provide :ros_bridge/OpenDRPose2D "88f7162365f7e82118b9fd3fc8f9ae3b")


