#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
       echo "ROS_DISTRO is not defined"
       exit 1
fi

MODULE_PATH=${OPENDR_HOME}/src/opendr/utils/eagerx
WS_PATH=${OPENDR_HOME}/lib/catkin_ws_eagerx

## ROS
sudo apt-get update && sudo apt-get install -y \
  ros-${ROS_DISTRO}-ros-base \
  python3-rosdep || exit;
source /opt/ros/${ROS_DISTRO}/setup.bash

## EAGERx
# Check if EAGERx repo is already cloned
if [ ! -d ${MODULE_PATH}/eagerx ]; then
  cd ${MODULE_PATH}}
  git clone -b opendr git@github.com:eager-dev/eagerx.git
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
