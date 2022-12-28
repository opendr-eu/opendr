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

## ROS
sudo apt-get update && sudo apt-get install -y \
  ros-${ROS_DISTRO}-ros-base \
  ros-${ROS_DISTRO}-pybind11-catkin \
  ros-${ROS_DISTRO}-moveit \
  ros-${ROS_DISTRO}-tf-conversions \
  ros-${ROS_DISTRO}-eigen-conversions \
  ros-${ROS_DISTRO}-pr2-controllers-msgs \
  ros-${ROS_DISTRO}-pr2-mechanism-msgs \
  ros-${ROS_DISTRO}-pr2-description \
  ros-${ROS_DISTRO}-gazebo-msgs \
  python3-rosdep || exit;
source /opt/ros/${ROS_DISTRO}/setup.bash

## packages to install from source
if [ ! -f ${WS_PATH}/mobile_manipulation_pr2.rosinstall ]; then
  mkdir -p ${WS_PATH}/src
  cd ${WS_PATH}
  sudo rosdep init
  rosdep update --rosdistro $ROS_DISTRO
  cp ${MODULE_PATH}/mobile_manipulation_pr2.rosinstall .
  vcs import src --input mobile_manipulation_pr2.rosinstall
  rosdep install -y -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3 tiago_pcl_tutorial"
  echo "update the moveit configs with the global joint"
fi
# tiago: will be added as soon as supporting ros noetic
#  && yes | rosinstall src /opt/ros/$ROS_DISTRO mobile_manipulation_tiago.rosinstall \
#  && cp ${MODULE_PATH}/robots_world/tiago/modified_tiago.srdf.em src/tiago_moveit_config/config/srdf/tiago.srdf.em

# build the catkin workspace
cd ${WS_PATH} || exit
# link the mobile_manipulation package into the ws
ln -s ${MODULE_PATH} src/
source /opt/ros/${ROS_DISTRO}/setup.bash
# will fail if we don't specify `-j x`
catkin build -j
# NOTE: users have to work in the shell that has sourced this file
source devel/setup.bash
