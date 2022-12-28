#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
  echo "OPENDR_HOME is not defined"
  exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
  echo "ROS_DISTRO is not defined"
  exit 1
fi

echo "Setup End-to-end Planning"

sudo sh -c 'echo deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

MODULE_PATH=${OPENDR_HOME}/src/opendr/planning/end_to_end_planning
WS_PATH=${OPENDR_HOME}/projects/opendr_ws/

# ROS
if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-webots-ros \
    ros-${ROS_DISTRO}-ros-numpy \
    python3-rosdep python3-empy || exit;

    git clone --branch $ROS_DISTRO https://github.com/ros-perception/vision_opencv $WS_PATH/src/vision_opencv

    source /opt/ros/${ROS_DISTRO}/setup.bash

    # build the catkin workspace
    cd ${WS_PATH} || exit
    source /opt/ros/${ROS_DISTRO}/setup.bash
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
fi

# ROS2
if [[ ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-webots-ros2 || exit;
fi
