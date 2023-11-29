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
sudo apt-get install --yes unzip libfreetype6-dev lsb-release git python3-pip curl wget python3.8-venv qt5-default

# Get all submodules
git submodule init
git submodule update

# Temporary fix
echo "[WARNING] Removing Panoptic Segmentation tool and Continual SLAM."
rm -rf ./src/opendr/perception/panoptic_segmentation
rm -rf ./src/opendr/perception/continual_slam

# Create a virtual environment and update
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install setuptools configparser wheel==0.38.4 lark

# Add repositories for ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
  && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Build OpenDR
make install_compilation_dependencies
make install_runtime_dependencies

# ROS package dependencies
if [[ ${ROS_DISTRO} == "noetic" || ${ROS_DISTRO} == "melodic" ]]; then
  echo "Installing ROS dependencies"
  sudo apt-get -y install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-geometry-msgs ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-audio-common-msgs ros-$ROS_DISTRO-hri-msgs ros-$ROS_DISTRO-usb-cam ros-$ROS_DISTRO-webots-ros
fi

# ROS2 package dependencies
if [[ ${ROS_DISTRO} == "foxy" || ${ROS_DISTRO} == "humble" ]]; then
  echo "Installing ROS2 dependencies"
  sudo apt-get -y install python3-lark ros-$ROS_DISTRO-usb-cam ros-$ROS_DISTRO-webots-ros2 python3-colcon-common-extensions ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-sensor-msgs-py
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/$ROS_DISTRO/lib/controller
  cd $OPENDR_HOME/projects/opendr_ws_2/
  git clone --depth 1 --branch ros2 https://github.com/ros-drivers/audio_common src/audio_common
  rosdep install -i --from-path src/audio_common --rosdistro $ROS_DISTRO -y
  cd $OPENDR_HOME
fi

# If working on GPU install GPU dependencies as needed
if [[ "${OPENDR_DEVICE}" == "gpu" ]]; then
  python3 -m pip uninstall -y mxnet
  python3 -m pip uninstall -y torch
  echo "[INFO] Replacing  mxnet-cu112==1.8.0post0 to enable CUDA acceleration."
  python3 -m pip install mxnet-cu112==1.8.0post0
  echo "[INFO] Replacing torch==1.13.1+cu116 to enable CUDA acceleration."
  python3 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  echo "[INFO] Reinstalling detectronv2."
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
  echo "[INFO] Installing TensorRT dependencies."
  python -m pip install tensorrt==8.6.1
  python -m pip install pycuda==2023.1
fi

make libopendr

deactivate
