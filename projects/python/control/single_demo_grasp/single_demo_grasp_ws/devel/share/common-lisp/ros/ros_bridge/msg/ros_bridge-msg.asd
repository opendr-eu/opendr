
(cl:in-package :asdf)

(defsystem "ros_bridge-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "OpenDRPose2D" :depends-on ("_package_OpenDRPose2D"))
    (:file "_package_OpenDRPose2D" :depends-on ("_package"))
    (:file "OpenDRPose2DKeypoint" :depends-on ("_package_OpenDRPose2DKeypoint"))
    (:file "_package_OpenDRPose2DKeypoint" :depends-on ("_package"))
  ))