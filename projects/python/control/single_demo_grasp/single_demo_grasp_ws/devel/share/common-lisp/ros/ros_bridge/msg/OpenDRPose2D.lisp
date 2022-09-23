; Auto-generated. Do not edit!


(cl:in-package ros_bridge-msg)


;//! \htmlinclude OpenDRPose2D.msg.html

(cl:defclass <OpenDRPose2D> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (pose_id
    :reader pose_id
    :initarg :pose_id
    :type cl:integer
    :initform 0)
   (conf
    :reader conf
    :initarg :conf
    :type cl:float
    :initform 0.0)
   (keypoint_list
    :reader keypoint_list
    :initarg :keypoint_list
    :type (cl:vector ros_bridge-msg:OpenDRPose2DKeypoint)
   :initform (cl:make-array 0 :element-type 'ros_bridge-msg:OpenDRPose2DKeypoint :initial-element (cl:make-instance 'ros_bridge-msg:OpenDRPose2DKeypoint))))
)

(cl:defclass OpenDRPose2D (<OpenDRPose2D>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OpenDRPose2D>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OpenDRPose2D)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ros_bridge-msg:<OpenDRPose2D> is deprecated: use ros_bridge-msg:OpenDRPose2D instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <OpenDRPose2D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:header-val is deprecated.  Use ros_bridge-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'pose_id-val :lambda-list '(m))
(cl:defmethod pose_id-val ((m <OpenDRPose2D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:pose_id-val is deprecated.  Use ros_bridge-msg:pose_id instead.")
  (pose_id m))

(cl:ensure-generic-function 'conf-val :lambda-list '(m))
(cl:defmethod conf-val ((m <OpenDRPose2D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:conf-val is deprecated.  Use ros_bridge-msg:conf instead.")
  (conf m))

(cl:ensure-generic-function 'keypoint_list-val :lambda-list '(m))
(cl:defmethod keypoint_list-val ((m <OpenDRPose2D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:keypoint_list-val is deprecated.  Use ros_bridge-msg:keypoint_list instead.")
  (keypoint_list m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OpenDRPose2D>) ostream)
  "Serializes a message object of type '<OpenDRPose2D>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let* ((signed (cl:slot-value msg 'pose_id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'conf))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'keypoint_list))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'keypoint_list))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OpenDRPose2D>) istream)
  "Deserializes a message object of type '<OpenDRPose2D>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'pose_id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'conf) (roslisp-utils:decode-single-float-bits bits)))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'keypoint_list) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'keypoint_list)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'ros_bridge-msg:OpenDRPose2DKeypoint))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OpenDRPose2D>)))
  "Returns string type for a message object of type '<OpenDRPose2D>"
  "ros_bridge/OpenDRPose2D")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OpenDRPose2D)))
  "Returns string type for a message object of type 'OpenDRPose2D"
  "ros_bridge/OpenDRPose2D")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OpenDRPose2D>)))
  "Returns md5sum for a message object of type '<OpenDRPose2D>"
  "88f7162365f7e82118b9fd3fc8f9ae3b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OpenDRPose2D)))
  "Returns md5sum for a message object of type 'OpenDRPose2D"
  "88f7162365f7e82118b9fd3fc8f9ae3b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OpenDRPose2D>)))
  "Returns full string definition for message of type '<OpenDRPose2D>"
  (cl:format cl:nil "# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message represents a full OpenDR human pose 2D as a list of keypoints~%~%Header header~%~%# The id of the pose~%int32 pose_id~%~%# The pose detection confidence of the model~%float32 conf~%~%# A list of a human 2D pose keypoints~%OpenDRPose2DKeypoint[] keypoint_list~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: ros_bridge/OpenDRPose2DKeypoint~%# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message contains all relevant information for an OpenDR human pose 2D keypoint~%~%# The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes~%string kpt_name~%~%# x and y pixel position on the input image, (0, 0) is top-left corner of image~%int32 x~%int32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OpenDRPose2D)))
  "Returns full string definition for message of type 'OpenDRPose2D"
  (cl:format cl:nil "# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message represents a full OpenDR human pose 2D as a list of keypoints~%~%Header header~%~%# The id of the pose~%int32 pose_id~%~%# The pose detection confidence of the model~%float32 conf~%~%# A list of a human 2D pose keypoints~%OpenDRPose2DKeypoint[] keypoint_list~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: ros_bridge/OpenDRPose2DKeypoint~%# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message contains all relevant information for an OpenDR human pose 2D keypoint~%~%# The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes~%string kpt_name~%~%# x and y pixel position on the input image, (0, 0) is top-left corner of image~%int32 x~%int32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OpenDRPose2D>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'keypoint_list) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OpenDRPose2D>))
  "Converts a ROS message object to a list"
  (cl:list 'OpenDRPose2D
    (cl:cons ':header (header msg))
    (cl:cons ':pose_id (pose_id msg))
    (cl:cons ':conf (conf msg))
    (cl:cons ':keypoint_list (keypoint_list msg))
))
