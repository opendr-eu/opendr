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

You can directly install OpenDR toolkit for CPU-only inference using pip.
First, install the required dependencies:
```bash
export DISABLE_BCOLZ_AVX2=true
sudo apt install python3.8-venv libfreetype6-dev git build-essential cmake python3-dev wget
python3 -m venv venv
source venv/bin/activate
wget https://raw.githubusercontent.com/opendr-eu/opendr/master/dependencies/pip_requirements.txt
cat pip_requirements.txt | xargs -n 1 -L 1 pip install
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
