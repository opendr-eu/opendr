#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
  echo "OPENDR_HOME is not defined"
  exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
  echo "ROS_DISTRO is not defined"
  exit 1
fi

echo "Setup Single Demonstration Grasp"

sudo sh -c 'echo deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

MODULE_PATH=${OPENDR_HOME}/src/opendr/control/single_demo_grasp
WS_PATH=${OPENDR_HOME}/projects/python/control/single_demo_grasp/simulation_ws
BRIDGE_PATH=${OPENDR_HOME}/projects/opendr_ws/src/ros_bridge

if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-libfranka \
    python3-empy python3-catkin-tools || exit;

  git clone --branch $ROS_DISTRO https://github.com/ros-perception/vision_opencv $WS_PATH/src/vision_opencv

  source /opt/ros/${ROS_DISTRO}/setup.bash

  # build the catkin workspace
  cd ${WS_PATH} || exit
  ln -s ${MODULE_PATH} src/
  ln -s ${BRIDGE_PATH} src/
  source /opt/ros/${ROS_DISTRO}/setup.bash
  catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
  source devel/setup.bash
fi