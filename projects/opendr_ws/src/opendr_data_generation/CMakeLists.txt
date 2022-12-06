cmake_minimum_required(VERSION 3.0.2)
project(opendr_data_generation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
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
   scripts/synthetic_facial_generation_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
