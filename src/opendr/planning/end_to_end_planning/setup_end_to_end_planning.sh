#!/bin/bash

if [[ -z "$ROS_DISTRO" ]]; then
  echo "ROS_DISTRO is not defined"
  exit 1
fi

sudo sh -c 'echo deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# ROS
if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-webots-ros \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-ros-numpy \
    python3-rosdep python3-empy || exit;
fi

# ROS2
if [[ ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-webots-ros2 || exit;
fi
