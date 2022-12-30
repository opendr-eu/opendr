# Customizing the toolkit

OpenDR is fully open-source and can be readily customized to meet the needs of several different application areas, since the source code for all the developed tools is provided.
Several ready-to-use examples, which are expected to cover a wide range of different needs, are provided.
For example, users can readily use the existing [ROS nodes](../../projects/opendr_ws), e.g., by including the required triggers or by combining several nodes into one to build custom nodes that will fit their needs. 
Furthermore, note that several tools can be combined within a ROS node, as showcased in [face recognition ROS node](../../projects/opendr_ws/src/perception/scripts/face_recognition.py). 
You can use these nodes as a template for customizing the toolkit to your own needs.
The rest of this document includes instructions for:
1. [Building docker images using the provided docker files](#building-custom-docker-images)
2. [Customizing existing docker images](#customizing-existing-docker-images)
3. [Changing the behavior of ROS nodes](#changing-the-behavior-of-ros-nodes)
4. [Building docker images that do not contain the whole toolkit](#building-docker-images-that-do-not-contain-the-whole-toolkit)


## Building custom docker images
The default docker images can be too large for some applications.
OpenDR provides the dockerfiles for customizing the images to your own needs, e.g., using OpenDR in custom third-party images.
Therefore, you can build the docker images locally using the [Dockerfile](/Dockerfile) ([Dockerfile-cuda](/Dockerfile-cuda) for cuda) provided in the root folder of the toolkit.

### Building the CPU image
For the CPU image, execute the following commands:
```bash
git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr
cd opendr
sudo docker build -t opendr/opendr-toolkit:cpu .
```

### Building the CUDA image
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

### Building the Embedded Devices image
The provided Dockerfile-embedded is tested on fresh flashed Nvidia-nx, Nvidia-Tx2 and Nvidia-Agx using jetpack 4.6.

To build the embedded devices images yourself, first edit `/etc/docker/daemon.json` in order to set the default docker runtime:
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

Then run:
```
sudo docker build --build-arg device=nx -t opendr/opendr-toolkit:nx -f Dockerfile-embedded .
```
You can build the image on nx/tx2/agx by changing the build-arg accordingly.

### Running the custom images
In order to run them, the commands are respectively:
```bash
sudo docker run -p 8888:8888 opendr/opendr-toolkit:cpu
```
or:
```
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cuda
```
or:
```
sudo docker run -p 8888:8888 opendr/opendr-toolkit:nx
```
## Customizing existing docker images
Building docker images from scratch can take a lot of time, especially for embedded systems without cross-compilation support.
If you need to modify a docker image without rebuilding it (e.g., for changing some source files inside it or adding support for custom pipelines), then you can simply start with the image that you are interesting in, make the changes and use the [docker commit](https://docs.docker.com/engine/reference/commandline/commit/) command. In this way, the changes that have been made will be saved in a new image.


## Changing the behavior of ROS nodes
ROS nodes are provided as examples that demonstrate how various tools can be used. 
As a result, customization might be needed in order to make them appropriate for your specific needs.
Currently, all nodes support changing the input/output topics (please refer to the [README](../../projects/opendr_ws/src/opendr_perception/README.md) for more information for each node).
However, if you need to change anything else (e.g., load a custom model), then you should appropriately modify the source code of the nodes.
This is very easy, since the Python API of OpenDR is used in all of the provided nodes.
You can refer to [Python API documentation](https://github.com/opendr-eu/opendr/blob/master/docs/reference/index.md) for more details for the tool that you are interested in.

### Loading a custom model
Loading a custom model in a ROS node is very easy. 
First, locate the node that you want to modify (e.g., [pose estimation](../../projects/opendr_ws/src/perception/scripts/pose_estimation.py)).
Then, search for the line where the learner loads the model (i.e., calls the `load()` function). 
For the aforementioned node, this happens at [line 76](../../projects/opendr_ws/src/perception/scripts/pose_estimation.py#L76).
Then, replace the path to the `load()` function with the path to your custom model.
You can also optionally remove the call to `download()` function (e.g., [line 75](../../projects/opendr_ws/src/perception/scripts/pose_estimation.py#L75)) to make the node start up faster. 


## Building docker images that do not contain the whole toolkit
To build custom docker images that do not contain the whole toolkit you should follow these steps:
1. Identify the tools that are using and note them.
2. Start from a clean clone of the repository and remove all modules under [src/opendr] that you are not using. 
To this end, use the `rm` command from the root folder of the toolkit and write down the commands that you are issuing.
Please note that you should NOT remove the `engine` package. 
4. Add the `rm` commands that you have issued in the dockerfile (e.g., in the main [dockerfile](https://github.com/opendr-eu/opendr/blob/master/Dockerfile)) after the `WORKDIR command` and before the `RUN ./bin/install.sh` command.
5. Build the dockerfile as usual.

By removing the tools that you are not using, you are also removing the corresponding `requirements.txt` file. 
In this way, the `install.sh` script will not pull and install the corresponding dependencies, allowing for having smaller and more lightweight docker images.

Things to keep in mind:
1. ROS noetic is manually installed by the installation script. 
If you want to install another version, you should modify both `install.sh` and `Makefile`.
2. `mxnet`, `torch` and `detectron` are manually installed by the `install.sh` script if you have set `OPENDR_DEVICE=gpu`.
If you do not need these dependencies, then you should manually remove them.
