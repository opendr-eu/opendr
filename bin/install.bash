#!/bin/bash
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export DISABLE_BCOLZ_AVX2=true

if [[ -z "${OPENDR_DEVICE}" ]]; then
  echo "[INFO] Set available device to GPU. You can manually change this by running 'export OPENDR_DEVICE=cpu'."
  export OPENDR_DEVICE=cpu
fi

# Install base ubuntu deps
sudo apt-get install --yes libfreetype6-dev lsb-release git python python3-pip curl wget

# Get all submodules
git submodule init
git submodule update

case $(lsb_release -r |cut -f2) in
  "18.04")
    export ROS_DISTRO=melodic;;
  "20.04")
    export ROS_DISTRO=noetic;;
  *)
    echo "Not tested for this ubuntu version" && exit 1;;
esac

# Create a virtual environment and update
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
python3 -m pip install -U pip
pip3 install setuptools

# Add repositories for ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
            && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Build OpenDR
make install_compilation_dependencies
make install_runtime_dependencies
make libopendr

# Prepare requirements.txt for wheel distributions
pip3 freeze > requirements.txt
python setup.py bdist

# Install OpenDR
pip3 install dist/*.whl

deactivate

