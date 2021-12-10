# Perception Package

This folder contains a catkin workspace to build the simulation package and its dependencies.

## Setup

The workspace will be setup by installing compilation and runtime dependencies when setting up the OpenDR toolkit:

```
$ make install_compilation_dependencies
$ make install_runtime_dependencies
```

after installing dependencies, the user must source the workspace in the shell in order to detect the packages:

```
$ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
```

also, the user need to set the environment variable below to find webots directory:
```
$ export WEBOTS_HOME=/usr/local/webots
```

## Demos

three different nodes must be launched consecutively in order to properly run the grasping pipeline:

1. first, the simulation environment must be loaded. open a new terminal and run the following commands:

```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ export WEBOTS_HOME=/usr/local/webots
5. $ roslaunch single_demo_grasping_demo panda_sim.launch 
```

2. secondly, open a second terminal and run camera stream node that runs inference on images from camera:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ roslaunch single_demo_grasping_demo camera_stream_inference.launch.launch 
```

3. finally, open a third terminal and run commander node to control the robot step by step:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ roslaunch single_demo_grasping_demo panda_sim_control.launch 
```

## Examples
You can find an example on how to use the learner class to run inference and see the result in the following directory:
```
$ cd projects/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
```
simply run:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ cd projects/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/ 
5. $ ./single_demo_inference.py
```

