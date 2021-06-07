cmake_minimum_required(VERSION 3.0.2)
project(opendr_example)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  eager_core
  eager_bridge_pybullet
  eager_bridge_webots
  eager_robot_ur5e
  eager_process_safe_actions
  rospy
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
   scripts/example_multi_robot.py
   scripts/example_safe_actions.py
   scripts/example_switch_engine.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
