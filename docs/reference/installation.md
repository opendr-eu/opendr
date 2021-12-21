# Installing OpenDR toolkit

OpenDR can be installed in the following ways:
1. By cloning this repository (CPU/GPU support)
2. Using *pip* (CPU only)
3. Using *docker* (CPU/GPU support)

The following table summarizes the installation options based on your system architecture and OS:

| Installation Method | CPU/GPU  | OS                    |
|---------------------|----------|-----------------------|
| Clone & Install     | Both     | Ubuntu 20.04 (x86-64) |
| pip                 | CPU-only | Ubuntu 20.04 (x86-64) |
| docker              | Both     | Linux                 |


# Installing by cloning OpenDR repository (Ubuntu 20.04, x86, architecture)

This is the recommended way of installing the toolkit, since it allows for fully exploiting all the provided functionalities.
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
Note that this might take a while (~0.1-1h depending on your machine), while the script also makes system-wide changes.
Using dockerfiles is strongly advised (please see below), unless you know what you are doing.
Please also make sure that you have enough RAM available for the installation.


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
You can also refer to this [Dockerfile](https://github.com/opendr-eu/opendr/blob/master/Dockerfile-cuda) for installation instructions.
Note that NVIDIA 30xx GPUs may not be fully supported, due to CUDA limitations.

# Installing using *pip*

You can directly install OpenDR toolkit for CPU-only inference using pip.
First, install the required dependencies:
```bash
export DISABLE_BCOLZ_AVX2=true
sudo apt install python3.8-venv libfreetype6-dev git build-essential cmake python3-dev
python3 -m venv venv
source venv/bin/activate
pip install Cython torch==1.7.1 wheel
pip install git+https://github.com/cidl-auth/cocoapi@03ee5a19844e253b8365dbbf35c1e5d8ca2e7281#subdirectory=PythonAPI
pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
pip install git+https://github.com/MatthewHowe/DCNv2@194f5733c667cf13e5bd478a8c5bf27573ffa98c
```
Then, you can install the Python API of the toolkit using pip:
```bash
pip install opendr-toolkit
```
For some systems it is necessary to disable AVX2 during `bcolz` dependency installation (`export DISABLE_BCOLZ_AVX2=true`).
Please note that only the Python API is exposed when you install OpenDR toolkit using *pip*.

*pip* wheels only install code that is available under the *src/opendr* folder of the toolkit.
Tools provided in *projects* are not installed by *pip*.

# Installing using *docker*
## CPU docker
After installing [docker](https://docs.docker.com/engine/install/ubuntu/), you can directly pull opendr CPU image as:
````bash
TODO
````
This docker automatically runs a Jupyter notebook server that listens at port 8888.
You can run this docker and map this port in your localhost as:
```bash
sudo docker run -p 8888:8888 opendr/ubuntu
```
You can build our docker container (based on Ubuntu 20.04) using the dockerfile provided in the root folder of the toolkit:
```bash
cd opendr
sudo docker build -t opendr/ubuntu .
```

## GPU docker
If you want to use a CUDA-enabled container please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Then, you can directly use opendr-gpu as:
```bash
TODO
sudo docker run -p 8888:8888 opendr/ubuntu
```
You can also build the image by yourself using the supplied dockerfile. 
First, edit `/etc/docker/daemon.json` in order to set the default docker runtime:
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia-container-runtime"
}
```
Restart docker afterwards:
```
sudo systemctl restart docker.service
```
Then you can build the supplied dockerfile:
```bash
cd opendr
sudo docker build -t opendr/ubuntu -f Dockerfile-cuda .
```
As before, you can run this docker:
```bash
sudo docker run --gpus all -p 8888:8888 opendr/ubuntu
```
