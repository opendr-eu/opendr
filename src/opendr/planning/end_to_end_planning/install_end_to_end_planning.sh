#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
  echo "OPENDR_HOME is not defined"
  exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
  echo "ROS_DISTRO is not defined"
  exit 1
fi

MODULE_PATH=${OPENDR_HOME}/src/opendr/control/mobile_manipulation
WS_PATH=${OPENDR_HOME}/projects/python/control/mobile_manipulation/mobile_manipulation_ws

# ROS
if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-webots-ros \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-$ROS_DISTRO-ros-numpy \
    python3-rosdep || exit;
fi

# ROS2
if [[ ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-$ROS_DISTRO-webots-ros2
fi

source /opt/ros/${ROS_DISTRO}/setup.bash
