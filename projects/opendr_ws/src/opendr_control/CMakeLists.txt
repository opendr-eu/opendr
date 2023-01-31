cmake_minimum_required(VERSION 3.0.2)
project(opendr_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  actionlib_msgs
  geometry_msgs
)

catkin_python_setup()


add_service_files(
  FILES
  Grasp.srv
  MoveGripper.srv
  RotateEE.srv
  SetJointState.srv
  SetPoseTarget.srv
  SetPoseTarget1D.srv
  SetPoseTarget2D.srv
)

add_action_files(
  FILES
  Pick.action
  Place.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS actionlib_msgs
)

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
   scripts/opendr_grasp_example.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
