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
2. If you are planning to use a usb camera for the demos, install the corresponding package and its dependencies (Note that usb_cam package cannot be installed on Nvidia embedded devices):
   
```shell
cd src
git clone https://github.com/ros-drivers/usb_cam
cd ..
rosdep install --from-paths src/ --ignore-src
```
3. Install the following dependencies, required in order to use the OpenDR ROS tools (On Nvidia embedded devices skip to 6.):
```shell
sudo apt-get install ros-noetic-vision-msgs ros-noetic-geometry-msgs ros-noetic-sensor-msgs ros-noetic-audio-common-msgs
```
4. Build the packages inside workspace
```shell
catkin_make
```
5. Source the workspace and you are ready to go!
```shell
source devel/setup.bash
```
6. Nvidia embedded devices: install vision_opencv:
```shell
cd ./projects/opendr_ws/src
git clone https://github.com/ros-perception/vision_opencv
cd ..
catkin_make
```
Then source the workspace:
```shell
source devel/setup.bash
```
## Structure

Currently, apart from tools, opendr_ws contains the following ROS nodes:

### [Perception](src/perception/README.md)
1. Pose Estimation
2. Fall Detection
3. 2D Object Detection
4. Face Detection
5. Panoptic Segmentation
6. Face Recognition
7. Semantic Segmentation
8. RGBD Hand Gesture Recognition
9. Heart Anomaly Detection
10. Video Human Activity Recognition
11. Landmark-based Facial Expression Recognition
12. Skeleton-based Human Action Recognition
13. Speech Command Recognition
14. Voxel Object Detection 3D
15. AB3DMOT Object Tracking 3D
16. FairMOT Object Tracking 2D
17. Deep Sort Object Tracking 2D
