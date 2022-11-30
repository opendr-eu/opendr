cmake_minimum_required(VERSION 3.0.2)
project(opendr_simulation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  shape_msgs
  vision_msgs
  message_generation
)

add_service_files(
  FILES
  Mesh_vc.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  shape_msgs
  vision_msgs
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
   scripts/human_model_generation_service.py
   scripts/human_model_generation_client.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
