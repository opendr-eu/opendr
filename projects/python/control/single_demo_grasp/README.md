# Perception Package

This folder contains a catkin workspace to build the simulation package and its dependencies.

## Workspace Setup

In order to run the demo, [Webots](https://cyberbotics.com/#download) simulator is required.
- download Webots 2021b for your platform from [here](https://github.com/cyberbotics/webots/releases/tag/R2021b) and install it
- install webots-ros, where ROS_DISTRO must be either `melodic` or `noetic`
```
$ sudo apt-get install ros-ROS_DISTRO-webots-ros
```
- set the environment variable below, by pointing to the location where Webots was installed.
In ubuntu you can do so by executing the following command in a terminal:
```
$ export WEBOTS_HOME=/usr/local/webots
```

The workspace will be setup by installing compilation and runtime dependencies when setting up the OpenDR toolkit. From the OpenDR folder, run:

```
$ make install_compilation_dependencies
$ make install_runtime_dependencies
```

After installing dependencies, the user must source the workspace in the shell in order to detect the packages:

```
$ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
```

## Demos

three different nodes must be launched consecutively in order to properly run the grasping pipeline:

1. first, the simulation environment must be loaded. open a new terminal and run the following commands:

```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ export WEBOTS_HOME=/usr/local/webots
5. $ roslaunch single_demo_grasping_demo panda_sim.launch
```

2. secondly, open a second terminal and run camera stream node that runs inference on images from camera:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ roslaunch single_demo_grasping_demo camera_stream_inference.launch
```

3. finally, open a third terminal and run commander node to control the robot step by step:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ roslaunch single_demo_grasping_demo panda_sim_control.launch
```

## Examples
You can find an example on how to use the learner class to run inference and see the result in the following directory:
```
$ cd projects/python/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
```
simply run:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ cd projects/python/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
5. $ ./single_demo_inference.py
```
