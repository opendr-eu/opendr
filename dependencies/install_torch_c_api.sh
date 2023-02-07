#!/bin/bash

if [[ -z "$TORCH_VERSION" ]];
then
  echo "Torch version not defined, version 1.9.0 will be installed."
  echo "For a specific Torch version please define TORCH_VERSION with 'export TORCH_VERSION=x.x.x'"
  TORCH_VERSION="1.9.0"
fi

if [ ! -f /usr/local/lib/libtorchvision.so ]; then
  TORCH_DIRECTORY="/usr/local/libtorch"

  if [[ "$OPENDR_DEVICE" == "gpu" ]]
  then
    echo "Downloading and installing LibTorch and torchvision (gpu support) ..."
    GPU="on"
    DEVICE="gpu"
    CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
  else
    echo "Downloading and installing LibTorch and torchvision (cpu-only) ..."
    GPU="off"
    DEVICE="cpu"
  fi

  # Find CUDA version and download torch and vision
  echo "Downloading LibTorch and torchvision ..."
  # Make sure that we can download files
  if [[ -z "$CUDA_PATH" ]];
  then
      python3 ./download_torch.py --opendr_device "$OPENDR_DEVICE" --torch_version "$TORCH_VERSION"
  else
      python3 ./download_torch.py --opendr_device "$OPENDR_DEVICE" --torch_version "$TORCH_VERSION" --cuda_path "$CUDA_PATH"
  fi
  echo "Downloading Libtorch and torchvision done."

  # TORCH INSTALLATION
  unzip -qq libtorch.zip
  cd libtorch

  sudo mkdir -p ${TORCH_DIRECTORY}
  sudo cp -r ./* ${TORCH_DIRECTORY}
  cd ..

  # TORCH VISION INSTALLATION
  tar zxf vision.tar.gz
  mv vision-* vision
  cd vision
  sudo mkdir -p build
  cd build
  sudo cmake .. -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} -DCMAKE_PREFIX_PATH=${TORCH_DIRECTORY} -DWITH_CUDA=${GPU}
  sudo make
  sudo make install
  cd ../..

  # CLEAN
  sudo rm -rf libtorch
  sudo rm -rf libtorch.zip

  sudo rm -rf vision
  sudo rm -rf vision.tar.gz

  sudo ldconfig

fi
