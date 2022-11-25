#!/bin/bash

if [ ! -f /usr/local/lib/libtorchvision.so ]; then
  CUDA_VERSION="116"
  TORCH_VERSION="1.13.0"
  TORCH_DIRECTORY="/usr/local/libtorch"

  VISION_VERSION="0.14.0"
  if [[ "$OPENDR_DEVICE" == "gpu" ]]
  then
      echo "Downloading and installing libtorch and torchvision (gpu support) ..."
      GPU="on"
      DEVICE="cu"${CUDA_VERSION}
      CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
  else
      echo "Downloading and installing libtorch and torchvsion (cpu-only) ..."
      GPU="off"
      DEVICE="cpu"
  fi

  # TORCH INSTALLATION
  wget https://download.pytorch.org/libtorch/${DEVICE}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${DEVICE}.zip --quiet
  unzip -qq libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+${DEVICE}.zip
  cd libtorch

  sudo mkdir -p ${TORCH_DIRECTORY}
  sudo cp -r ./* ${TORCH_DIRECTORY}
  cd ..

  # TORCH VISION INSTALLATION
  wget https://github.com/pytorch/vision/archive/refs/tags/v${VISION_VERSION}.tar.gz --quiet
  tar zxf v${VISION_VERSION}.tar.gz
  cd vision-${VISION_VERSION}
  sudo mkdir -p build
  cd build
  sudo cmake .. -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} -DCMAKE_PREFIX_PATH=${TORCH_DIRECTORY} -DWITH_CUDA=${GPU}
  sudo make
  sudo make install
  cd ../..

  # CLEAN
  sudo rm -rf libtorch
  sudo rm -rf libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+${DEVICE}.zip

  sudo rm -rf vision-${VISION_VERSION}
  sudo rm -rf v${VISION_VERSION}.tar.gz

  sudo ldconfig

fi

