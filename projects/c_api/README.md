# OpenDR C API Demos

C API comes with a number of demo applications inside the `samples` folder that demonstrate the usage of the API.
You can build them and try them (check the `build` folder for the executables) using the following command:
```sh
make demos
```
Make sure that you have downloaded the necessary resources before running the demo (`make download`) and that you execute the binaries from the root folder of the C API. 

## Supported tools
Currently, the following tools are exposing a C API:
1. Activity recognition (X3D)
2. Face recognition
3. Pose estimation (Lightweight OpenPose)
3. Object detection 2D (DETR)
4. Object detection 2D (NanoDet)
5. Object tracking 2D (DeepSort)
6. Skeleton-based action recognition (Progressive Spatiotemporal GCN)

