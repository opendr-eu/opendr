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
WS_PATH=${OPENDR_HOME}/lib/catkin_ws_mobile_manipulation

# ROS
sudo apt-get update && sudo apt-get install \
  ros-${ROS_DISTRO}-ros-base \
  ros-${ROS_DISTRO}-pybind11-catkin \
  ros-${ROS_DISTRO}-moveit \
  ros-${ROS_DISTRO}-pr2-simulator \
  ros-${ROS_DISTRO}-moveit-pr2
source /opt/ros/${ROS_DISTRO}/setup.bash

# libgp
LIBGP_PATH=${OPENDR_HOME}/lib/libgp
if [ ! -d ${LIBGP_PATH} ]; then
  git clone --single-branch --depth 1 https://github.com/mblum/libgp.git ${LIBGP_PATH} \
    && cd ${LIBGP_PATH} \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF .. \
    && make \
    && sudo make install
fi

# tiago
if [ ! -f ${WS_PATH}/tiago_public.rosinstall ]; then
  mkdir ${WS_PATH} \
    && cd ${WS_PATH} \
    && wget -O tiago_public.rosinstall https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/kinetic-devel/tiago_public-melodic.rosinstall \
    && yes | rosinstall src /opt/ros/$ROS_DISTRO tiago_public.rosinstall \
    && rosdep init \
    && rosdep update --rosdistro $ROS_DISTRO \
    && sudo apt-get update \
    && rosdep install -y -r -q --from-paths src --ignore-src --rosdistro $ROS_DISTRO --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3 tiago_pcl_tutorial" \
    && echo "update the moveit configs with the global joint" \
    && cp ${MODULE_PATH}/robots_world/tiago/modified_tiago.srdf.em src/tiago_moveit_config/config/srdf/tiago.srdf.em \
    && cp ${MODULE_PATH}/robots_world/tiago/modified_tiago_pal-gripper.srdf src/tiago_moveit_config/config/srdf/tiago_pal-gripper.srdf \
    && cp ${MODULE_PATH}/robots_world/tiago/modified_gripper.urdf.xacro src/pal_gripper/pal_gripper_description/urdf/gripper.urdf.xacro \
    && cp ${MODULE_PATH}/robots_world/tiago/modified_wsg_gripper.urdf.xacro src/pal_wsg_gripper/pal_wsg_gripper_description/urdf/gripper.urdf.xacro
fi

# build the catkin workspace
PYTHON_EXECUTABLE="$(which python3)"
PYTHON_INCLUDE_DIR="$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")"
PYTHON_LIBRARY="$(python3 -c 'from distutils import sysconfig; print("/".join([sysconfig.get_config_var("LIBPL"), sysconfig.get_config_var("LDLIBRARY")]))')"
cd ${WS_PATH}
ln -s ${MODULE_PATH} src
catkin config -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} --blacklist tiago_pcl_tutorial
source /opt/ros/${ROS_DISTRO}/setup.bash
# will fail if we don't specify `-j x`
catkin build -j 8
# NOTE: users have to work in the shell that has sourced this file
source devel/setup.bash
