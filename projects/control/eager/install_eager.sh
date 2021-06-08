#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
       echo "ROS is not installed"
       exit 1
fi

MODULE_PATH=${OPENDR_HOME}/projects/control/eager
OPENDR_WS=${OPENDR_HOME}/projects/opendr_ws

# install dependencies one by one to prevent interdependency errors
if [ -f "python_dependencies.txt" ]; then
       cat python_dependencies.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install
fi

source /opt/ros/$ROS_DISTRO/setup.bash
rosdep update --rosdistro $ROS_DISTRO

# install universal robot packages UR5 robot package
if [ ! -d ${OPENDR_WS}/src/universal_robot ]; then
    cd ${OPENDR_WS}/src \
    && git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git
fi

# install universal robot packages UR5 robot package
if [ ! -d ${OPENDR_WS}/src/ur_modern_driver ]; then
    cd ${OPENDR_WS}/src \
    && git clone -b kinetic-devel https://github.com/ros-industrial/ur_modern_driver.git
fi

# install eager
ln -s ${MODULE_PATH}/eager ${OPENDR_WS}/src

cd ${OPENDR_WS}
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
