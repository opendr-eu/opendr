cmake_minimum_required(VERSION 3.0.2)
project(opendr_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  vision_msgs
  std_msgs
  audio_common_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package()

###########
## Build ##
###########

include_directories(
    ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
   scripts/performance_node.py
   scripts/active_face_recognition_node.py
   scripts/pose_estimation_node.py
   scripts/hr_pose_estimation_node.py
   scripts/fall_detection_node.py
   scripts/wave_detection_node.py
   scripts/object_detection_2d_nanodet_node.py
   scripts/object_detection_2d_yolov5_node.py
   scripts/object_detection_2d_detr_node.py
   scripts/object_detection_2d_gem_node.py
   scripts/semantic_segmentation_bisenet_node.py
   scripts/semantic_segmentation_yolov8_node.py
   scripts/binary_high_resolution_node.py
   scripts/continual_slam_dataset_node.py
   scripts/continual_slam_predictor_node.py
   scripts/continual_slam_learner_node.py
   scripts/facial_emotion_estimation_node.py
   scripts/continual_skeleton_based_action_recognition_node.py
   scripts/point_cloud_2_publisher_node.py
   scripts/intent_recognition_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
