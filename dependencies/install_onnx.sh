#!/bin/bash

if [ ! -f /usr/local/lib/libonnxruntime.so ]; then

  
  VERSION="1.6.0"
  if [[ "$OPENDR_DEVICE" == "gpu" ]]
  then
      echo "Downloading and installing onnxruntime (gpu support) ..."
      DEVICE="-gpu"
  else
      echo "Downloading and installing onnxruntime (cpu-only) ..."
  fi

  wget https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64${DEVICE}-${VERSION}.tgz --quiet
  tar zxf onnxruntime-linux-x64${DEVICE}-${VERSION}.tgz
  cd onnxruntime-linux-x64${DEVICE}-${VERSION}
  sudo mkdir -p /usr/local/include/onnxruntime
  sudo cp include/* /usr/local/include/onnxruntime/
  sudo cp lib/libonnxruntime.so* /usr/local/lib/
  cd ..
  rm -rf onnxruntime-linux-x64${DEVICE}-${VERSION}
  rm -rf onnxruntime-linux-x64${DEVICE}-${VERSION}.tgz

  sudo ldconfig

fi
