#!/bin/bash

if [[ $1 = "tx2" ]];
then
  echo "Installing OpenDR on Nvidia TX2"
elif [[ $1 = "agx" ]] ||  [[ $1 = "nx" ]]
then
  echo "Installing OpenDR on Nvidia AGX/NX"
else
  echo "Wrong argument, supported inputs are 'tx2', 'agx' and 'nx'"
  exit 1
fi

# export OpenDR related paths
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/src:$LD_LIBRARY_PATH

# Install mxnet
cd $OPENDR_HOME

sudo apt-get install -y gfortran build-essential git python3-pip python-numpy libopencv-dev graphviz libopenblas-dev libopenblas-base libatlas-base-dev python-numpy

pip3 install --upgrade pip
pip3 install setuptools==59.5.0
pip3 install numpy==1.19.4

git clone --recursive -b v1.8.x https://github.com/apache/incubator-mxnet.git mxnet

export PATH=/usr/local/cuda/bin:$PATH
export MXNET_HOME=$OPENDR_HOME/mxnet/
export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda

cd $MXNET_HOME
cp $MXNET_HOME/make/config_jetson.mk config.mk
sed -i 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk
sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/' config.mk
# CUDA_ARCH setting
sed -i 's/CUDA_ARCH = -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72/ /' config.mk
sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk

if [[ $1 = "tx2" ]];
then
  sed -i '/USE_CUDNN/a CUDA_ARCH = -gencode arch=compute_62,code=sm_62' config.mk
elif [[ $1 = "agx" ]] ||  [[ $1 = "nx" ]]
then
  echo "AGX or nx"
  sed -i '/USE_CUDNN/a CUDA_ARCH = -gencode arch=compute_72,code=sm_72' config.mk
else
  echo "Wrong argument, supported inputs are 'tx2', 'agx' and 'nx'"
fi

make -j $(nproc) NVCC=/usr/local/cuda/bin/nvcc

cd $MXNET_HOME/python
sudo pip3 install -e .

cd $OPENDR_HOME
chmod a+rwx ./mxnet

sudo apt-get update
sudo apt-get install --yes libfreetype6-dev lsb-release  curl wget

git submodule init
git submodule update

pip3 install configparser

# Install Torch
sudo apt-get install --yes libopenblas-dev cmake ninja-build
TORCH=torch-1.9.0-cp36-cp36m-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl

pip3 install Cython
pip3 install $TORCH
rm ./torch-1.9.0-cp36-cp36m-linux_aarch64.whl

# Install Torchvision
TORCH_VISION=0.10.0
sudo apt-get install --yes libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.10.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.10.0
sudo python3 setup.py install
cd ../
rm -r torchvision/

# Install dlib
wget http://dlib.net/files/dlib-19.21.tar.bz2
tar jxvf dlib-19.21.tar.bz2
cd dlib-19.21/
mkdir build
cd build/
cmake ..
cmake --build .
cd ../
sudo python3 setup.py install
cd $OPENDR_HOME
rm dlib-19.21.tar.bz2

apt-get install -y libprotobuf-dev protobuf-compiler
apt-get install -y python3-tk
# For AV
apt-get update && apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:jonathonf/ffmpeg-4

apt-get update && apt-get install -y \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libeigen3-dev

pip3 install av==8.0.1

# Install rest of the dependencies of OpenDR

pip3 install absl-py==1.0.0
pip3 install aiohttp==3.8.1
pip3 install aiosignal==1.2.0
pip3 install alembic==1.7.5
pip3 install appdirs==1.4.4
pip3 install async-timeout==4.0.1
pip3 install attrs==21.2.0
pip3 install audioread==2.1.9
pip3 install autocfg==0.0.8
pip3 install Automat==20.2.0
pip3 install autopage==0.4.0
pip3 install bcolz==1.2.1
pip3 cache purge
pip3 install scikit-build==0.16.3
pip3 install cachetools==4.2.4
pip3 install catkin-pkg==0.4.24
pip3 install catkin-tools==0.8.2
pip3 install certifi==2021.10.8
pip3 install cityscapesscripts==2.2.0
pip3 install charset-normalizer==2.0.9
pip3 install cliff==3.10.0
pip3 install cloudpickle==1.5.0
pip3 install cmaes==0.8.2
pip3 install cmd2==2.3.3
pip3 install colorlog==6.6.0
pip3 install configparser==5.2.0
pip3 install constantly==15.1.0
pip3 install cycler==0.11.0
pip3 install Cython==0.29.22
pip3 install cython-bbox==0.1.3
pip3 install decorator==5.1.0
pip3 install defusedxml==0.7.1
pip3 install distro==1.6.0
pip3 install docutils==0.18.1
pip3 install easydict==1.9
pip3 install empy==3.3.4
pip3 install filterpy==1.4.5
pip3 install flake8==4.0.1
pip3 install flake8-import-order==0.18.1
pip3 install flask
pip3 cache purge
pip3 install frozenlist==1.2.0
pip3 install fsspec==2021.11.1
pip3 install future==0.18.2
pip3 install gdown
pip3 install gluoncv==0.11.0b20210908
pip3 install google-auth==1.35.0
pip3 install google-auth-oauthlib==0.4.6
pip3 install graphviz==0.8.4
pip3 install greenlet==1.1.2
pip3 install grpcio==1.42.0
pip3 install gym==0.21.0
pip3 install hyperlink==21.0.0
pip3 install idna==3.3
pip3 install idna-ssl==1.1.0
pip3 install imageio==2.6.0
pip3 install imantics==0.1.12
pip3 install imgaug==0.4.0
pip3 install importlib-metadata==4.8.2
pip3 install importlib-resources==5.4.0
pip3 install imutils==0.5.4
pip3 install incremental==21.3.0
pip3 install iniconfig==1.1.1
pip3 install ipython
pip3 install joblib==1.0.1
pip3 install kiwisolver==1.3.1
pip3 install lap==0.4.0
pip3 cache purge
sudo apt-get install --yes llvm-10*
sudo ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config
pip3 install llvmlite==0.36.0
sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.h.bak
pip3 install numba==0.53.1
LLVM_CONFIG=/usr/bin/llvm-config-10 pip3 install librosa==0.8.0
pip3 install lxml==4.6.3
pip3 install Mako==1.1.6
pip3 install Markdown==3.3.6
pip3 install MarkupSafe==2.0.1
pip3 install matplotlib==2.2.2
pip3 install mccabe==0.6.1
pip3 install mmcv==0.5.9
pip3 install motmetrics==1.2.0
pip3 install multidict==5.2.0
pip3 install munkres==1.1.4
pip3 install netifaces==0.11.0
pip3 install networkx==2.5.1
pip3 install numpy==1.19.4
pip3 install oauthlib==3.1.1
pip3 install onnx==1.10.2
pip3 install onnxruntime==1.3.0
pip3 install opencv-python==4.5.4.60
pip3 install opencv-contrib-python==4.5.4.60
pip3 cache purge
pip3 install optuna==2.10.0
pip3 install osrf-pycommon==1.0.0
pip3 install packaging==21.3
pip3 install pandas==1.1.5
pip3 install pbr==5.8.0
pip3 install Pillow==8.3.2
pip3 install plotly==5.4.0
pip3 install pluggy==1.0.0
pip3 install pooch==1.5.2
pip3 install portalocker==2.3.2
pip3 install prettytable==2.4.0
pip3 install progress==1.5
pip3 install protobuf==3.19.6
pip3 install py==1.11.0
pip3 install py-cpuinfo==8.0.0
pip3 install pyasn1==0.4.8
pip3 install pyasn1-modules==0.2.8
pip3 install pybind11==2.6.2
pip3 install pycodestyle==2.8.0
pip3 install pycparser==2.21
pip3 install pyflakes==2.4.0
pip3 install pyglet==1.5.16
pip3 install pyparsing==3.0.6
pip3 install pyperclip==1.8.2
pip3 install pytest==6.2.5
pip3 install pytest-benchmark==3.4.1
pip3 install python-dateutil==2.8.2
pip3 cache purge
pip3 install pytz==2021.3
pip3 install PyWavelets==1.1.1
pip3 install --ignore-installed PyYAML==5.3
pip3 install requests==2.26.0
pip3 install requests-oauthlib==1.3.0
pip3 install resampy==0.2.2
pip3 install rosdep==0.21.0
pip3 install rosdistro==0.8.3
pip3 install roslibpy==1.2.1
pip3 install rospkg==1.3.0
pip3 install rsa==4.8
pip3 install scikit-image==0.16.2
pip3 install scikit-learn==0.22
pip3 install seaborn==0.11.2
pip3 install setuptools-rust==1.1.2
pip3 install scipy==1.5.4
pip3 install Shapely==1.5.9
pip3 install six==1.16.0
pip3 install SoundFile==0.10.3.post1
pip3 install SQLAlchemy==1.4.28
pip3 install stable-baselines3==1.1.0
pip3 install stevedore==3.5.0
pip3 install tabulate==0.8.9
pip3 install tenacity==8.0.1
pip3 install tensorboard==2.4.1
pip3 install tensorboard-plugin-wit==1.8.0
pip3 install tensorboardX==2.0
pip3 cache purge
pip3 install toml==0.10.2
pip3 install tqdm==4.54.0
pip3 install trimesh==3.5.23
pip3 install Twisted==21.7.0
pip3 install txaio==21.2.1
pip3 install typing_extensions==4.0.1
pip3 install urllib3==1.26.7
pip3 install vcstool==0.3.0
pip3 install wdwidth==0.2.5
pip3 install Werkzeug==2.0.2
pip3 install xmljson==0.2.1
pip3 install xmltodict==0.12.0
pip3 install yacs==0.1.8
pip3 install yarl==1.7.2
pip3 install zipp==3.6.0
pip3 install zope.interface==5.4.0
pip3 install wheel
pip3 install pytorch-lightning==1.2.3
pip3 install omegaconf==2.3.0
pip3 install ninja
pip3 install terminaltables
pip3 install psutil
pip3 install continual-inference>=1.0.2
pip3 install git+https://github.com/waspinator/pycococreator.git@0.2.0
pip3 install git+https://github.com/cidl-auth/cocoapi@03ee5a19844e253b8365dbbf35c1e5d8ca2e7281#subdirectory=PythonAPI
pip3 install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
pip3 install git+https://github.com/mapillary/inplace_abn.git
pip3 install git+https://github.com/facebookresearch/detectron2.git@4841e70ee48da72c32304f9ebf98138c2a70048d
pip3 install git+https://github.com/cidl-auth/DCNv2
pip3 install ${OPENDR_HOME}/src/opendr/perception/panoptic_segmentation/efficient_ps/algorithm/EfficientPS
pip3 install ${OPENDR_HOME}/src/opendr/perception/panoptic_segmentation/efficient_ps/algorithm/EfficientPS/efficientNet
pip3 cache purge

cd $OPENDR_HOME/src/opendr/perception/object_detection_2d/retinaface
make
cd $OPENDR_HOME
