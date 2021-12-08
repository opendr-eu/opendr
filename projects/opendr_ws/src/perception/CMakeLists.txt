cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  vision_msgs
  std_msgs
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
   scripts/pose_estimation.py
   scripts/object_detection_2d_detr.py
   scripts/object_detection_2d_gem.py
   scripts/semantic_segmentation_bisenet.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
