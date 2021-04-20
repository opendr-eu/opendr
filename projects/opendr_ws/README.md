# opendr_ws

## Description
This ROS workspace contains ROS nodes and tools developed by OpenDR project. Currently, ROS nodes are compatible with ROS Noetic.
This workspace contains the `ros_bridge` package, which provides message definitions for ROS-compatible OpenDR data types, 
as well the `ROSBridge` class which provides an interface to convert OpenDR data types and targets into ROS-compatible
ones similar to CvBridge. You can find more information in the corresponding [documentation](../../docs/reference/rosbridge.md).


## Setup
For running a minimal working example you can follow the instructions below:

0. Source the necessary distribution tools:
   
   ```source /opt/ros/noetic/setup.bash```
   
1. Make sure you are inside opendr_ws
2. If you are planning to use a usb camera for the demos, install the corresponding package:

```shell
cd src
git clone https://github.com/ros-drivers/usb_cam
cd ..
```
3. Build the packages inside workspace
```shell
catkin_make
```
4. Source the workspace and you are ready to go!
```shell
source devel/setup.bash
```
## Structure

Currently, apart from tools, opendr_ws contains the following 1 ros node:
1. [Pose Estimation](src/perception/README.md)
