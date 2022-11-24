cmake_minimum_required(VERSION 3.0.2)
project(control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
)

add_service_files(
  FILES
  RotateEE.srv
  SetJointState.srv
  SetPoseTarget.srv
  SetPoseTarget1D.srv
  SetPoseTarget2D.srv
)

generate_messages()

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
   scripts/moveit_grasp_control.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
