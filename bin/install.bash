#!/bin/sh
export OPENDR_HOME=$PWD
export OPENDR_DEVICE=cpu
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export DISABLE_BCOLZ_AVX2=true

# Get all submodules
git submodule init
git submodule update

# Create a virtual environment and update
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
python3 -m pip install -U pip
pip3 install setuptools

case $(lsb_release -r |cut -f2) in
  "18.04")
    export ROS_DISTRO=melodic;;
  "20.04")
    export ROS_DISTRO=noetic;;
  *)
    echo "Not tested for this ubuntu version" && exit 1;;
esac

# Add repositories for ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
            && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

make install_compilation_dependencies
make install_runtime_dependencies
make install libopendrl
