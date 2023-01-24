# Installing OpenDR toolkit

OpenDR can be installed in the following ways:
1. Using *pip* (CPU/GPU support)
2. Using *docker* (CPU/GPU support)
3. By cloning this repository (CPU/GPU support, for advanced users only)

The following table summarizes the installation options based on your system architecture and OS:

| Installation Method   | OS                    |
|-----------------------|-----------------------|
| Clone & Install       | Ubuntu 20.04 (x86-64) |
| pip                   | Ubuntu 20.04 (x86-64) |
| docker                | Linux / Windows       |

Note that pip installation includes only the Python API of the toolkit.
If you need to use all the functionalities of the toolkit (e.g., ROS nodes, etc.), then you need either to use the pre-compiled docker images or to follow the installation instructions for cloning and building the toolkit.

The toolkit is developed and tested on *Ubuntu 20.04 (x86-64)*.
Please make sure that you have the most recent version of all tools by running
```bash
sudo apt upgrade
```
before installing the toolkit and then follow the installation instructions in the relevant section.
All the required dependencies will be automatically installed (or explicit instructions are provided).
Other platforms apart from Ubuntu 20.04, e.g., Windows, other Linux distributions, etc., are currently supported through docker images.

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
For example, if you stick with the default PyTorch version (1.8) and use CUDA11.2, then you can simply follow:
```bash
sudo apt install python3.8-venv libfreetype6-dev git build-essential cmake python3-dev wget libopenblas-dev libsndfile1 libboost-dev libeigen3-dev
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install mxnet-cu112==1.8.0post0
pip install opendr-toolkit-engine
pip install opendr-toolkit
```
If you encounter any issue installing the latest version of detectron, then you can try installing a previous commit:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
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
- *opendr-toolkit-activity-recognition*
- *opendr-toolkit-speech-recognition*
- *opendr-toolkit-semantic-segmentation*
- *opendr-toolkit-skeleton-based-action-recognition*
- *opendr-toolkit-face-recognition*
- *opendr-toolkit-facial-expression-recognition*
- *opendr-toolkit-panoptic-segmentation*
- *opendr-toolkit-pose-estimation*
- *opendr-toolkit-compressive-learning*
- *opendr-toolkit-hyperparameter-tuner*
- *opendr-toolkit-heart-anomaly-detection*
- *opendr-toolkit-human-model-generation*
- *opendr-toolkit-multimodal-human-centric*
- *opendr-toolkit-object-detection-2d*
- *opendr-toolkit-object-tracking-2d*
- *opendr-toolkit-object-detection-3d*
- *opendr-toolkit-object-tracking-3d*
- *opendr-toolkit-ambiguity-measure*
- *opendr-toolkit-fall-detection*

Note that `opendr-toolkit` is actually just a metapackage that includes all the aformentioned packages.


# Installing using *docker*
## CPU docker
After installing [docker](https://docs.docker.com/engine/install/ubuntu/), you can directly run the OpenDR image as:
```bash
sudo docker run -p 8888:8888 opendr/opendr-toolkit:cpu_v2.0.0
```
The docker automatically runs a Jupyter notebook server that listens at port 8888.
When launched, you can access the Jupyter notebook by following the link provided in the console, it should be similar to [http://127.0.0.1:8888/?token=TOKEN](http://127.0.0.1:8888/?token=TOKEN). In order to stop the container, please quit the Jupyter notebook.

If you do not wish to use Jupyter, you can also experiment by starting an interactive session by running:
```bash
sudo docker run -it opendr/opendr-toolkit:cpu_v2.0.0 /bin/bash
```
In this case, do not forget to enable the virtual environment with:
```bash
source bin/activate.sh
```
If you want to display GTK-based applications from the Docker container (e.g., visualize results using OpenCV `imshow()`), then you should mount the X server socket inside the container, e.g.,
```bash
xhost +local:root
sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY opendr/opendr-toolkit:cpu_v2.0.0 /bin/bash
```

## GPU docker
If you want to use a CUDA-enabled container please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Then, you can directly run the latest image with the command:
```bash
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cuda_v2.0.0
```
or, for an interactive session:
```bash
sudo docker run --gpus all -it opendr/opendr-toolkit:cuda_v2.0.0 /bin/bash
```
In this case, do not forget to enable the virtual environment with:
```bash
source bin/activate.sh
```

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

If you want to install GPU-related dependencies, then you can appropriately set the `OPENDR_DEVICE` variable.
The toolkit defaults to using CPU.
Therefore, if you want to use GPU, please set this variable accordingly *before* running the installation script:
```bash
export OPENDR_DEVICE=gpu
```

If you want to use ROS or ROS2, then you need to set the `ROS_DISTRO` variable *before* running the installation script so that additional required dependencies are correctly installed.
This variable should be set to either `noetic` or `melodic` for ROS, and `foxy` or `humble` for ROS2.

You are then ready to install the toolkit:
```bash
cd opendr
./bin/install.sh
```
The installation script automatically installs all the required dependencies.
Note that this might take a while (~10-20min depending on your machine and network connection), while the script also makes system-wide changes.
Using dockerfiles is strongly advised (please see below), unless you know what you are doing.
Please also make sure that you have enough RAM available for the installation (about 4GB of free RAM is needed for the full installation/compilation).


The installation script creates a *virtualenv*, where the toolkit is installed.
To activate OpenDR environment you can just source the `activate.sh`:
```bash
source ./bin/activate.sh
```
Then, you are ready to use the toolkit!

**NOTE:** `OPENDR_DEVICE` does not alter the inference/training device at *runtime*.
It only affects the dependency installation.
You can use OpenDR API to change the inference device.

You can also verify the installation by using the supplied Python and C unit tests:
```bash
make unittest
make ctests
```

If you plan to use GPU-enabled functionalities, then you are advised to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive), along with [CuDNN](https://developer.nvidia.com/cudnn).

**HINT:** All tests probe for the `TEST_DEVICE` enviromental variable when running.
If this enviromental variable is set during testing, it allows for easily running all tests on a different device (e.g., setting `TEST_DEVICE=cuda:0` runs all tests on the first GPU of the system).


## Nvidia embedded devices docker
You can also run the corresponding docker image on an Nvidia embedded device (supported: TX-2, Xavier-NX and AGX):

Note that the embedded device should be flashed with Jetpack 4.6.

To enable GPU usage on the embedded device within docker, first edit `/etc/docker/daemon.json` in order to set the default docker runtime:
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


You can directly run the corresponding docker image by running one of the below:
```bash
sudo docker run -it opendr/opendr-toolkit:tx2_v2 /bin/bash
sudo docker run -it opendr/opendr-toolkit:nx_v2 /bin/bash
sudo docker run -it opendr/opendr-toolkit:agx_v2 /bin/bash
```
This will give you access to a bash terminal within the docker.

After that you should enable the environment variables inside the docker with:
```bash
cd opendr
source bin/activate_nvidia.sh
source /opt/ros/noetic/setup.bash
source projects/opendr_ws/devel/setup.bash
```

The embedded devices docker comes preinstalled with the OpenDR toolkit.
It supports all tools under perception package, as well as all corresponding ROS nodes.

You can enable a USB camera, given it is mounted as `/dev/video0`,  by running the container with the following arguments:
```
xhost +local:root
sudo docker run -it --privileged -v /dev/video0:/dev/video0 opendr/opendr-toolkit:nx_v2 /bin/bash
```

To use the docker on an embedded device with a monitor and a usb camera attached, as well as network access through the hosts network settings you can run:
```
xhost +local:root
sudo docker run -it --privileged --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DSIPLAY -v /dev/video0:/dev/video0 opendr/opendr-toolkit:nx_v2 /bin/bash
```