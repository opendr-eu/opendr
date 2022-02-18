# Installing OpenDR toolkit

OpenDR can be installed in the following ways:
1. By cloning this repository (CPU/GPU support)
2. Using *pip* (CPU/GPU support)
3. Using *docker* (CPU/GPU support)

The following table summarizes the installation options based on your system architecture and OS:

| Installation Method | CPU/GPU  | OS                    |
|---------------------|----------|-----------------------|
| Clone & Install     | Both     | Ubuntu 20.04 (x86-64) |
| pip                 | Both     | Ubuntu 20.04 (x86-64) |
| docker              | Both     | Linux / Windows       |


# Installing by cloning OpenDR repository (Ubuntu 20.04, x86, architecture)

This is the recommended way of installing the whole toolkit, since it allows for fully exploiting all the provided functionalities.
To install the toolkit, please first make sure that you have `git` available on your system.
```bash
sudo apt install git
```
Then, clone the toolkit:
```bash
git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr
```
You are then ready to install the toolkit:
```bash
cd opendr
./bin/install.sh
```


The installation script automatically installs all the required dependencies.
Note that this might take a while (~10-20min depending on your machine and network connection), while the script also makes system-wide changes.
Using dockerfiles is strongly advised (please see below), unless you know what you are doing.
Please also make sure that you have enough RAM available for the installation (about 4GB of free RAM is needed for the full installation/compilation).

You can set the inference/training device using the `OPENDR_DEVICE` variable.
The toolkit defaults to using CPU.
If you want to use GPU, please set this variable accordingly:
```bash
export OPENDR_DEVICE=gpu
```
The installation script creates a *virtualenv*, where the toolkit is installed.
To activate OpenDR environment you can just source the `activate.sh`:
```bash
source ./bin/activate.sh
```
Then, you are ready to use the toolkit!

You can also verify the installation by using the supplied Python and C unit tests:
```bash
make unittest
make ctests
```

If you plan to use GPU-enabled functionalities, then you are advised to install [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive).
To do so, you can follow these steps:
```bash
sudo apt install gcc-8 g++-8 gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
echo "Please switch to GCC 8"
sudo update-alternatives --config gcc
```
Then, you can install CUDA, along CuDNN.
You can also refer to this [dockerfile](https://github.com/opendr-eu/opendr/blob/master/Dockerfile-cuda) for installation instructions.
Note that NVIDIA 30xx GPUs may not be fully supported, due to CUDA limitations.

# Installing by cloning OpenDR repository on a Nvidia-TX2
If you are installing the toolkit on a Nvidia-TX2:
```bash
cd opendr
./bin/install_tx2.sh
```
Note that TX2 should be flashed with Jetpack 4.6 and that this might take a while (~4-5h), while the script also makes system-wide changes and might prompt for password input.

In order to use the toolkit, you should export some system variables by running `activate_tx2.sh`, (Note that this should be run before every toolkit use):
```bash
./bin/activate_tx2.sh
```
Then, you are ready to use the toolkit!

The script works also on Nvidia Xavier NX and AGX. You should change line 37 in [install_tx2.sh](https://github.com/opendr-eu/opendr/blob/tx2_install/bin/install_tx2.sh) from:
```
-gencode arch=compute_62,code=sm_62
```
to:
```
-gencode arch=compute_72,code=sm_72 
```
for Xavier NX and AGX Xavier.

# Installing using *pip*

## CPU-only installation

You can directly install the Python API of the OpenDR toolkit using pip.
First, install the required dependencies:
```bash
sudo apt install python3.8-venv libfreetype6-dev git build-essential cmake python3-dev wget libopenblas-dev libsndfile1 libboost-dev libeigen3-dev 
python3 -m venv venv
source venv/bin/activate
pip install wheel
```
Then, you  install the Python API of the toolkit using pip:
```bash
export DISABLE_BCOLZ_AVX2=true
pip install opendr-toolkit-engine
pip install opendr-toolkit
```
*pip* wheels only install code that is available under the *src/opendr* folder of the toolkit.
Tools provided in *projects* are not installed by *pip*.
If you have a CPU that does not support AVX2, the please also `export DISABLE_BCOLZ_AVX2=true` before installing the toolkit.
This is not needed for newer CPUs.

## Enabling GPU-acceleration
The same OpenDR package is used for both CPU and GPU systems. 
However, you need to have the appropriate GPU-enabled dependencies installed to use a GPU with OpenDR.
If you plan to use GPU, then you should first install [mxnet-cuda](https://mxnet.apache.org/versions/1.4.1/install/index.html?platform=Linux&language=Python&processor=CPU) and [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
For example, if you stick with the default PyTorch version (1.7) and use CUDA10.2, then you can simply follow:
```bash
sudo apt install python3.8-venv libfreetype6-dev git build-essential cmake python3-dev wget libopenblas-dev libsndfile1 libboost-dev libeigen3-dev 
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
pip install mxnet-cu102==1.8.0 
pip install opendr-toolkit-engine
pip install opendr-toolkit
```

## Installing only a *particular* tool using *pip* (CPU/GPU)

If you do not want to install the whole repository, you can only install a specific OpenDR tool.
For example, if you just want to perform pose estimation you can just run:
```bash
pip install opendr-toolkit-engine
pip install opendr-toolkit-pose-estimation
```
Note that `opendr-toolkit-engine` must be always installed in your system, while multiple tools can be installed in this way. 
OpenDR distributes the following packages that can be installed:
- *opendr-toolkit-activity_recognition*
- *opendr-toolkit-speech_recognition*
- *opendr-toolkit-semantic_segmentation*
- *opendr-toolkit-skeleton_based_action_recognition*
- *opendr-toolkit-face_recognition*
- *opendr-toolkit-facial_expression_recognition*
- *opendr-toolkit-panoptic_segmentation*
- *opendr-toolkit-pose_estimation*
- *opendr-toolkit-compressive_learning*
- *opendr-toolkit-hyperparameter_tuner*
- *opendr-toolkit-heart_anomaly_detection*
- *opendr-toolkit-human_model_generation*
- *opendr-toolkit-multimodal_human_centric*
- *opendr-toolkit-object_detection_2d*
- *opendr-toolkit-object_tracking_2d*
- *opendr-toolkit-object_detection_3d*
- *opendr-toolkit-object_tracking_3d*
- *opendr-toolkit-mobile_manipulation* (requires a functional ROS installation)
- *opendr-toolkit-single_demo_grasp* (requires a functional ROS installation)


Note that `opendr-toolkit` is actually just a metapackage that includes all the afformentioned packages.


# Installing using *docker*
## CPU docker
After installing [docker](https://docs.docker.com/engine/install/ubuntu/), you can directly run the OpenDR image as:
```bash
sudo docker run -p 8888:8888 opendr/opendr-toolkit:cpu_latest
```
The docker automatically runs a Jupyter notebook server that listens at port 8888.
When launched, you can access the Jupyter notebook by following the link provided in the console, it should be similar to [http://127.0.0.1:8888/?token=TOKEN](http://127.0.0.1:8888/?token=TOKEN). In order to stop the container, please quit the Jupyter notebook.

If you do not wish to use Jupyter, you can also experiment by starting an interactive session by running:
```bash
sudo docker run -it opendr/opendr-toolkit:cpu_latest /bin/bash
```
In this case, do not forget to enable the virtual environment with:
```bash
source bin/activate.sh
```
If you want to display GTK-based applications from the Docker container (e.g., visualize results using OpenCV `imshow()`), then you should mount the X server socket inside the container, e.g.,
```bash
xhost +local:root
sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY opendr/opendr-toolkit:cpu_latest /bin/bash
```

## GPU docker
If you want to use a CUDA-enabled container please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Then, you can directly run the latest image with the command:
```bash
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cuda_latest
```
or, for an interactive session:
```bash
sudo docker run --gpus all -it opendr/opendr-toolkit:cuda_latest /bin/bash
```
In this case, do not forget to enable the virtual environment with:
```bash
source bin/activate.sh
```
## Build the docker images yourself _(optional)_
Alternatively you can also build the docker images locally using the [Dockerfile](/Dockerfile) ([Dockerfile-cuda](/Dockerfile-cuda) for cuda) provided in the root folder of the toolkit.

For the CPU image, execute the following commands:
```bash
git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr
cd opendr
sudo docker build -t opendr/opendr-toolkit:cpu .
```

For the cuda-enabled image, first edit `/etc/docker/daemon.json` in order to set the default docker runtime:
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Restart docker afterwards:
```
sudo systemctl restart docker.service
```
Then you can build the supplied dockerfile:
```bash
git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr
cd opendr
sudo docker build -t opendr/opendr-toolkit:cuda -f Dockerfile-cuda .
```

In order to run them, the commands are respectively:
```bash
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cpu
```
and
```
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cuda
```
