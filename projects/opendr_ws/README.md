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
2. If you are planning to use a usb camera for the demos, install the corresponding package and its dependencies:

```shell
cd src
git clone https://github.com/ros-drivers/usb_cam
cd ..
rosdep install --from-paths src/ --ignore-src
```
3. Install the following dependencies, required in order to use the OpenDR ROS tools:
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
## Structure

Currently, apart from tools, opendr_ws contains the following ROS nodes (categorized according to the input they receive):

### [Perception](src/perception/README.md)
## RGB input
1. [Pose Estimation](src/perception/README.md#pose-estimation-ros-node)
2. [Fall Detection](src/perception/README.md#fall-detection-ros-node)
3. [Face Recognition](src/perception/README.md#face-recognition-ros-node)
4. [2D Object Detection](src/perception/README.md#2d-object-detection-ros-nodes)
5. [Face Detection](src/perception/README.md#face-detection-ros-node)
6. [Panoptic Segmentation](src/perception/README.md#panoptic-segmentation-ros-node)
7. [Semantic Segmentation](src/perception/README.md#semantic-segmentation-ros-node)
8. [Video Human Activity Recognition](src/perception/README.md#human-action-recognition-ros-node)
9. [Landmark-based Facial Expression Recognition](src/perception/README.md#landmark-based-facial-expression-recognition-ros-node)
10. [FairMOT Object Tracking 2D](src/perception/README.md#fairmot-object-tracking-2d-ros-node)
11. [Deep Sort Object Tracking 2D](src/perception/README.md#deep-sort-object-tracking-2d-ros-node)
12. [Skeleton-based Human Action Recognition](src/perception/README.md#skeleton-based-human-action-recognition-ros-node)
## Point cloud input
1. [Voxel Object Detection 3D](src/perception/README.md#voxel-object-detection-3d-ros-node)
2. [AB3DMOT Object Tracking 3D](src/perception/README.md#ab3dmot-object-tracking-3d-ros-node)
## RGB + Infrared input
1. [End-to-End Multi-Modal Object Detection (GEM)](src/perception/README.md#gem-ros-node)
## RGBD input nodes
1. [RGBD Hand Gesture Recognition](src/perception/README.md#rgbd-hand-gesture-recognition-ros-node)
## Biosignal input
1. [Heart Anomaly Detection](src/perception/README.md#heart-anomaly-detection-ros-node)
## Audio input
1. [Speech Command Recognition](src/perception/README.md#speech-command-recognition-ros-node)
