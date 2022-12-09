cmake_minimum_required(VERSION 3.0.2)
project(opendr_bridge)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  vision_msgs
  message_generation
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################
add_message_files(
    DIRECTORY msg
    FILES
    OpenDRPose2DKeypoint.msg
    OpenDRPose2D.msg
)

 add_service_files(
    DIRECTORY srv
    FILES
    OpenDRSingleObjectTracking.srv
 )

generate_messages(
    DEPENDENCIES
    std_msgs
    vision_msgs
 )

###################################
## catkin specific configuration ##
###################################

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs
)

###########
## Build ##
###########

include_directories(
    ${catkin_INCLUDE_DIRS}
)
