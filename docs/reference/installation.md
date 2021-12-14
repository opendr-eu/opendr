# Installing OpenDR toolkit

OpenDR can be installed in the following ways:
1. By cloning this repository (CPU/GPU support)
2. Using *pip* (CPU only)
3. Using *docker* (CPU only)

The following table summarizes the installation option based on your system architecture and OS:

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
git clone https://github.com/tasostefas/opendr_internal
```
You are then ready to install the toolkit:
```bash
cd opendr_internal
./bin/install.sh
```
The installation script automatically install all the required dependecies.
Note that you can set the inference device using the `OPENDR_DEVICE` variable.
The toolkit defaults to using CPU. If you want to use GPU, please set this variable accordingly:
```bash
export export OPENDR_DEVICE=cpu
```
The installation script creates a *virtualenv*, where the toolkit is installed.
To activate OpenDR environment you can just source the `activate.sh`:
```bash
source ./bin/activate.sh
```
Then, you are ready to use the toolkit!

# Installing using pip
You can directly install OpenDR toolkit for CPU-only inference using pip:
```bash
pip3 install opendr-eu
```
