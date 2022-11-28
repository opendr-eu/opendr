#!/bin/bash

if [[ -z "$TORCH_VERSION" ]];
then
  echo "Specific Torch Version is not defined. Torch version 1.9.0 will be installed"
  echo "For specific Torch Version plz defined variable TORCH_VERSION with export TORCH_VERSION=x.x.x."
  TORCH_VERSION="1.9.0"
fi

if [ ! -f /usr/local/lib/libtorchvision.so ]; then
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

  # Find CUDA version and download torch and vision
  echo "Downloading Libtorch and torchvision ..."
  # Make sure that we can download files
  if [[ -z "$CUDA_PATH" ]];
  then
      python3 ./download_torch.py --opendr_device "$OPENDR_DEVICE" --torch_version "$TORCH_VERSION"
  else
      python3 ./download_torch.py --opendr_device "$OPENDR_DEVICE" --torch_version "$TORCH_VERSION" --cuda_path "$CUDA_PATH"
  fi
  echo "Downloading Libtorch and torchvision ... FINIS"

  # TORCH INSTALLATION
  unzip -qq libtorch.zip
  cd libtorch

  sudo mkdir -p ${TORCH_DIRECTORY}
  sudo cp -r ./* ${TORCH_DIRECTORY}
  cd ..

  # TORCH VISION INSTALLATION
  tar zxf vision.tar.gz
  cd vision-${VISION_VERSION}
  sudo mkdir -p build
  cd build
  sudo cmake .. -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} -DCMAKE_PREFIX_PATH=${TORCH_DIRECTORY} -DWITH_CUDA=${GPU}
  sudo make
  sudo make install
  cd ../..

  # CLEAN
  sudo rm -rf libtorch
  sudo rm -rf libtorch.zip

  sudo rm -rf vision-${VISION_VERSION}
  sudo rm -rf vision.tar.gz

  sudo ldconfig

fi
