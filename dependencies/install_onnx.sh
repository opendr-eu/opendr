if [ ! -f /usr/local/lib/libonnxruntime.so ]; then

  echo "Downloading onnxruntime (this might take a while) ..."
  git clone -q --recursive https://github.com/Microsoft/onnxruntime
  cd onnxruntime

  # Checkout v1.6.0 (https://github.com/microsoft/onnxruntime/releases/tag/v1.6.0)
  git checkout v1.6.0 > /dev/null 2>&1

  echo "Building onnxruntime (this might take a while) ..."
  ./build.sh --config Release --build_shared_lib --parallel --skip_tests > /dev/null 2>&1

  echo "Installing onnxruntime..."
  sudo cp include/onnxruntime /usr/local/include/ -r
  sudo cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/
  sudo ldconfig

  echo "Cleaning up..."
  cd ..
  rm -rf onnxruntime
fi
