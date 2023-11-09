#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
  echo "OPENDR_HOME is not defined"
  exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
  echo "ROS_DISTRO is not defined"
  exit 1
fi

MODULE_PATH=${OPENDR_HOME}/src/opendr/control/single_demo_grasp
WS_PATH=${OPENDR_HOME}/projects/python/control/single_demo_grasp/simulation_ws
BRIDGE_PATH=${OPENDR_HOME}/projects/opendr_ws/src/ros_bridge

## Moveit
sudo apt install ros-${ROS_DISTRO}-moveit

## franka_ros libfranka
sudo apt install ros-${ROS_DISTRO}-libfranka

## build the catkin workspace
cd ${WS_PATH} || exit
ln -s ${MODULE_PATH} src/
ln -s ${BRIDGE_PATH} src/
source /opt/ros/${ROS_DISTRO}/setup.bash
catkin_make
source devel/setup.bash
