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

catkin_python_setup()

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
