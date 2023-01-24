# OpenDR C API

## Description

This module contains a C API that can be used for performing inference on models trained using the Python API of OpenDR.
Therefore, to use the C API you should first use the Python API to export a pretrained or a newly trained model and export it to ONNX or JIT format using the `optimize()` method.

## Setup

In order to build the library you should:

1. Build the OpenDR C API library

```sh
make libopendr
```

After building the API you will find the library (`libopendr.so`) under the `lib` folder.

## Using the C API

To use the C API in your applications you just need to include the header files (`include` folder) in your project and then link the compiled library by adding the following flags to gcc/g++, e.g.,
```
gcc you_program.c -o your_binary -I./include -L./lib -lopendr
```
