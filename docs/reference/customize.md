# Customizing the toolkit

OpenDR is fully open-source and can be readily customized to meet the needs of several different application areas, since the source code for all the developed tools is provided.
Several ready-to-use examples, which are expected to cover a wide range of different needs, are provided.
For example, users can readily use the existing [ROS nodes](projects/opendr_ws), e.g., by including the required triggers or by combining several nodes into one to build custom nodes that will fit their needs. 
The rest of this document includes instructions for:
1. Building docker images using the provided docker files


## Building custom docker images
The default docker images can be too large for some applications.
OpenDR provides the dockerfiles for customizing the images to your own needs.
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

### Running the custom images
In order to run them, the commands are respectively:
```bash
sudo docker run -p 8888:8888 opendr/opendr-toolkit:cpu
```
and
```
sudo docker run --gpus all -p 8888:8888 opendr/opendr-toolkit:cuda
```
