#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
       echo "ROS is not installed"
       exit 1
fi

MODULE_PATH=${OPENDR_HOME}/src/opendr/control/gym_tools/eager
UR_WS=${OPENDR_HOME}/lib/ur_ws
EAGER_WS=${OPENDR_HOME}/lib/eager_ws

. /opt/ros/${ROS_DISTRO}/setup.bash
rosdep update --rosdistro $ROS_DISTRO

# install universal robot packages UR5 robot package
if [ ! -f ${UR_WS}/devel/setup.bash ]; then
    mkdir -p ${UR_WS}/src \
    && cd ${UR_WS}/src \
    && git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git \
    && git clone -b kinetic-devel https://github.com/ros-industrial/ur_modern_driver.git \
    && cd ${UR_WS} \
    && rosdep install --rosdistro ${ROS_DISTRO} --ignore-src --from-paths src \
    && catkin_make
fi

. ${UR_WS}/devel/setup.bash

# install eager
if [ ! -d ${EAGER_WS}/src ]; then
    mkdir -p ${EAGER_WS}/src \
    && ln -s ${MODULE_PATH}/eager ${EAGER_WS}/src
fi

cd ${EAGER_WS}
rosdep install --rosdistro ${ROS_DISTRO} --ignore-src --from-paths src
catkin_make
echo ${EAGER_WS}
. ${EAGER_WS}/devel/setup.bash
