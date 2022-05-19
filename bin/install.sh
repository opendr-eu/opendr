#!/bin/bash
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export DISABLE_BCOLZ_AVX2=true

if [[ -z "${OPENDR_DEVICE}" ]]; then
  echo "[INFO] Set available device to CPU. You can manually change this by running 'export OPENDR_DEVICE=gpu'."
  export OPENDR_DEVICE=cpu
fi

# Install base ubuntu deps
sudo apt-get install --yes libfreetype6-dev lsb-release git python3-pip curl wget python3.8-venv

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

# If working on GPU install GPU dependencies as needed
if [[ "${OPENDR_DEVICE}" == "gpu" ]]; then
  pip3 uninstall -y mxnet
  pip3 uninstall -y torch
  echo "[INFO] Replacing  mxnet-cu112==1.8.0post0 to enable CUDA acceleration."
  pip3 install mxnet-cu112==1.8.0post0
  echo "[INFO] Replacing torch==1.9.0+cu111 to enable CUDA acceleration."
  pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  echo "[INFO] Reinstalling detectronv2."
  pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
fi

make libopendr

deactivate
