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


If you want to install GPU-related dependencies, then you can appropriately set the `OPENDR_DEVICE` variable.
The toolkit defaults to using CPU.
Therefore, if you want to use GPU, please set this variable accordingly *before* running the installation script:
```bash
export OPENDR_DEVICE=gpu
```
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

