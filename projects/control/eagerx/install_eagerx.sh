#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
       echo "ROS_DISTRO is not defined"
       exit 1
fi

MODULE_PATH=${OPENDR_HOME}/projects/control/eagerx
WS_PATH=${MODULE_PATH}/eagerx_ws

## ROS
sudo apt-get update && sudo apt-get install -y \
  ros-${ROS_DISTRO}-ros-base \
  python3-rosdep || exit;
source /opt/ros/${ROS_DISTRO}/setup.bash

# Check if EAGERx submodule is initialized
if [ ! -d ${MODULE_PATH}/eagerx/eagerx_core ]; then
  STR=$'EAGERx submodule not initialized\nPlease run:\ngit submodule init\ngit submodule update'
  echo "$STR"
  exit 1
fi

## packages to install from source
if [ ! -f ${WS_PATH}/ ]; then
  mkdir -p ${WS_PATH}/src
  cd ${WS_PATH}
  sudo rosdep init
  rosdep update --rosdistro $ROS_DISTRO
  rosdep install -y -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO
fi

# build the catkin workspace
cd ${WS_PATH}/src || exit
# link the eagerx package into the ws
if [ -f ${MODULE_PATH}/eagerx/opendr_package_list.txt ]; then
  while IFS= read -r line; do
    ln -s ${MODULE_PATH}/eagerx/$line;
  done < ${MODULE_PATH}/eagerx/opendr_package_list.txt
fi
source /opt/ros/${ROS_DISTRO}/setup.bash
cd ${WS_PATH} || exit
catkin build -j
# NOTE: users have to work in the shell that has sourced this file
source devel/setup.bash
