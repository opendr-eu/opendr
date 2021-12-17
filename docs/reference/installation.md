# Installing OpenDR toolkit

OpenDR can be installed in the following ways:
1. By cloning this repository (CPU/GPU support)
2. Using *pip* (CPU only)
3. Using *docker* (CPU only)

The following table summarizes the installation options based on your system architecture and OS:

| Installation Method | CPU/GPU  | OS                    |
|---------------------|----------|-----------------------|
| Clone & Install     | Both     | Ubuntu 20.04 (x86-64) |
| pip                 | CPU-only | Ubuntu 20.04 (x86-64) |
| docker              | CPU-only | Linux                 |

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
Note that you can set the inference device using the `OPENDR_DEVICE` variable.
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

# Installing using *pip*

You can directly install OpenDR toolkit for CPU-only inference using pip:
```bash
sudo apt install python3.8-venv libfreetype6-dev git
python3 -m venv venv
source venv/bin/activate
pip install Cython torch==1.7.1 wheel
pip install opendr-eu
```
Please note that only the Python API is exposed when you install OpenDR toolkit using *pip*.
*Note: This mode of installation is not yet available, since wheel is not yet published in PyPI.*

*pip* wheels only install code that is available under the *src/opendr* folder of the toolkit.
Tools provided in *projects* are not installed by *pip*.

# Installing using *docker*

After installing [docker](https://docs.docker.com/engine/install/ubuntu/), you can build our docker contrainer (based on Ubuntu 20.04):
```bash
sudo docker build -t opendr/ubuntu .
```
This docker automatically runs a Jupyter notebook server that listens at port 8888.
You can run this docker and map this port in you localhost as:
```bash
sudo docker run -p 8888:8888 opendr/ubuntu
```

