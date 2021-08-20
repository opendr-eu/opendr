# OpenDR C API

## Description

This module contains a C API that can be used for performing inference on models trained using the Python API of OpenDR. 
Therefore, to use the C API you should first use the Python API to train a model and then export it to ONNX format using the `optimize()` method.

## Setup
Before using the C API you should have the appropriate version of ONNX runtime installed. 
You can use the provided script (`install_deps.sh`) to install it, along with a number of other required dependencies.
Alternatively, you can also use the supplied Makefile that takes care both of the installation and compilation.
Please note that only Ubuntu 20.04 is currently officially supported.

In order to build the library you should:

0. Install the required binary dependencies:

```sh
make install_dependencies

1. Build the static library

```sh
make liboperdr

2. (Optionally) Download the necessary resources and verify that everything works as expected:

```sh
make download
make runtests
```
After building the API you will find the static library (`libopendr.a`) under the `lib` folder.

## Using the C API

To use the C API in your applications you just need to include the header files (`include` folder) in your project and then link the compiled library by adding the following flags to gcc/g++, e.g.,
```
gcc you_program.c -o your_binary -I./include -L./lib -lopendr
```

## Demo Applications
C API comes with a number of demo applications inside the `samples` folder that demonstrate the usage of the API.
You can build them and try them (check the `build` folder for the executables) using the following command:
```
makefile demos
```
Make sure that you have downloaded the necessary resources before running the demo (`makefile download`) and that you execute the binaries from the root folder of the C API. 

## Supported tools
Currently, the following tools are exposing a C API:
1. Face recognition
