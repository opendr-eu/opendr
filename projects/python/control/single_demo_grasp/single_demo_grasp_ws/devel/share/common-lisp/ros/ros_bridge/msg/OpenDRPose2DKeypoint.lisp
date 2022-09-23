; Auto-generated. Do not edit!


(cl:in-package ros_bridge-msg)


;//! \htmlinclude OpenDRPose2DKeypoint.msg.html

(cl:defclass <OpenDRPose2DKeypoint> (roslisp-msg-protocol:ros-message)
  ((kpt_name
    :reader kpt_name
    :initarg :kpt_name
    :type cl:string
    :initform "")
   (x
    :reader x
    :initarg :x
    :type cl:integer
    :initform 0)
   (y
    :reader y
    :initarg :y
    :type cl:integer
    :initform 0))
)

(cl:defclass OpenDRPose2DKeypoint (<OpenDRPose2DKeypoint>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OpenDRPose2DKeypoint>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OpenDRPose2DKeypoint)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ros_bridge-msg:<OpenDRPose2DKeypoint> is deprecated: use ros_bridge-msg:OpenDRPose2DKeypoint instead.")))

(cl:ensure-generic-function 'kpt_name-val :lambda-list '(m))
(cl:defmethod kpt_name-val ((m <OpenDRPose2DKeypoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:kpt_name-val is deprecated.  Use ros_bridge-msg:kpt_name instead.")
  (kpt_name m))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <OpenDRPose2DKeypoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:x-val is deprecated.  Use ros_bridge-msg:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <OpenDRPose2DKeypoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ros_bridge-msg:y-val is deprecated.  Use ros_bridge-msg:y instead.")
  (y m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OpenDRPose2DKeypoint>) ostream)
  "Serializes a message object of type '<OpenDRPose2DKeypoint>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'kpt_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'kpt_name))
  (cl:let* ((signed (cl:slot-value msg 'x)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'y)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OpenDRPose2DKeypoint>) istream)
  "Deserializes a message object of type '<OpenDRPose2DKeypoint>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'kpt_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'kpt_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'x) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'y) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OpenDRPose2DKeypoint>)))
  "Returns string type for a message object of type '<OpenDRPose2DKeypoint>"
  "ros_bridge/OpenDRPose2DKeypoint")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OpenDRPose2DKeypoint)))
  "Returns string type for a message object of type 'OpenDRPose2DKeypoint"
  "ros_bridge/OpenDRPose2DKeypoint")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OpenDRPose2DKeypoint>)))
  "Returns md5sum for a message object of type '<OpenDRPose2DKeypoint>"
  "6ce75e74f73663ed82a3235764bc7edf")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OpenDRPose2DKeypoint)))
  "Returns md5sum for a message object of type 'OpenDRPose2DKeypoint"
  "6ce75e74f73663ed82a3235764bc7edf")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OpenDRPose2DKeypoint>)))
  "Returns full string definition for message of type '<OpenDRPose2DKeypoint>"
  (cl:format cl:nil "# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message contains all relevant information for an OpenDR human pose 2D keypoint~%~%# The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes~%string kpt_name~%~%# x and y pixel position on the input image, (0, 0) is top-left corner of image~%int32 x~%int32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OpenDRPose2DKeypoint)))
  "Returns full string definition for message of type 'OpenDRPose2DKeypoint"
  (cl:format cl:nil "# Copyright 2020-2022 OpenDR European Project~%#~%# Licensed under the Apache License, Version 2.0 (the \"License\");~%# you may not use this file except in compliance with the License.~%# You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS,~%# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.~%# See the License for the specific language governing permissions and~%# limitations under the License.~%~%# This message contains all relevant information for an OpenDR human pose 2D keypoint~%~%# The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes~%string kpt_name~%~%# x and y pixel position on the input image, (0, 0) is top-left corner of image~%int32 x~%int32 y~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OpenDRPose2DKeypoint>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'kpt_name))
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OpenDRPose2DKeypoint>))
  "Converts a ROS message object to a list"
  (cl:list 'OpenDRPose2DKeypoint
    (cl:cons ':kpt_name (kpt_name msg))
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
))
