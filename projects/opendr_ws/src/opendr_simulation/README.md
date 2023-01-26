# OpenDR Simulation Package

This package contains ROS nodes related to simulation package of OpenDR.

## Human Model Generation ROS Node
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can 


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
```

2. You can start the human model generation service node. 

```shell
rosrun opendr_simulation human_model_generation_service.py
```

3. An example client node can run to examine the basic utilities of the service.
```shell
rosrun opendr_simulation human_model_generation_client.py
```
