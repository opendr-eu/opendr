#!/bin/bash
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export DISABLE_BCOLZ_AVX2=true

if [[ -z "${OPENDR_DEVICE}" ]]; then
  echo "[INFO] Set available device to CPU. You can manually change this by running 'export OPENDR_DEVICE=gpu'."
  export OPENDR_DEVICE=cpu
fi

if [[ -z "${ROS_DISTRO}" ]]; then
  echo "[INFO] No ROS_DISTRO is specified. The modules relying on ROS/ROS2 might not work."
else
  if ! ([[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" || ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]); then
    echo "[ERROR] ${ROS_DISTRO} is not a supported ROS_DISTRO. Please use 'noetic' or 'melodic' for ROS and 'foxy' or 'humble' for ROS2."
    exit 1
  fi
fi

# Install base ubuntu deps
sudo apt-get install --yes libfreetype6-dev lsb-release git python3-pip curl wget python3.8-venv

# Get all submodules
git submodule init
git submodule update

# Create a virtual environment and update
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
pip3 install setuptools configparser

# Add repositories for ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
            && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Build OpenDR
make install_compilation_dependencies
make install_runtime_dependencies

# Install additional ROS packages
if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  echo "Installing ROS dependencies"
  sudo apt-get -y install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-geometry-msgs ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-audio-common-msgs ros-$ROS_DISTRO-usb-cam
fi

# Install additional ROS2 packages
if [[ ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]; then
  echo "Installing ROS2 dependencies"
  sudo apt-get -y install ros-$ROS_DISTRO-usb-cam
fi

# If working on GPU install GPU dependencies as needed
if [[ "${OPENDR_DEVICE}" == "gpu" ]]; then
  pip3 uninstall -y mxnet
  pip3 uninstall -y torch
  echo "[INFO] Replacing  mxnet-cu112==1.8.0post0 to enable CUDA acceleration."
  pip3 install mxnet-cu112==1.8.0post0
  echo "[INFO] Replacing torch==1.9.0+cu111 to enable CUDA acceleration."
  pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  echo "[INFO] Reinstalling detectronv2."
  pip3 install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
fi

make libopendr

deactivate
