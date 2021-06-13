echo "Installing dependencies..."
sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev

echo "Getting onnxruntime..."

if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
  echo "ONNX installation not found!"
  pip3 install flake8 --upgrade
  git clone --recursive https://github.com/Microsoft/onnxruntime
  cd onnxruntime

  # Checkout v1.6.0 (https://github.com/microsoft/onnxruntime/releases/tag/v1.6.0)
  git checkout 718ca7f

  echo "Building onnxruntime..."
  ./build.sh --config Release --build_shared_lib --parallel

  echo "Installing onnxruntime..."
  sudo cp include/onnxruntime /usr/local/include/ -r
  sudo cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/
  sudo ldconfig

  echo "Cleaning up..."
  cd ..
  rm -rf onnxruntime
fi


