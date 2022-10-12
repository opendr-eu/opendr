# opendr_ws_2

## Description
This ROS2 workspace contains ROS2 nodes and tools developed by OpenDR project. Currently, ROS2 nodes are compatible with ROS2 Foxy.
This workspace contains the `opendr_ros2_bridge` package, which contains the `ROS2Bridge` class that provides an interface to convert OpenDR data types and targets into ROS-compatible
ones similar to CvBridge. The workspace also contains the `opendr_ros2_interfaces` which provides message and service definitions for ROS-compatible OpenDR data types. You can find more information in the corresponding [opendr_ros2_bridge documentation](../../docs/reference/ros2bridge.md) and [opendr_ros2_interfaces documentation](). <!-- add interfaces readme link -->

## Setup

For running a minimal working example you can follow the instructions below:

1. Make sure that [ROS2-foxy is installed.](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
2. Source the necessary distribution tools:
    ```shell
    source /opt/ros/foxy/setup.bash
    ```
   _For convenience, you can add this line to your `.bashrc` so you don't have to source the tools each time you open a  terminal window._
3. Install some dependencies:
    ```shell
    # Install colcon https://docs.ros.org/en/foxy/Tutorials/Colcon-Tutorial.html
    sudo apt install python3-colcon-common-extensions
    # Install vision messages
    sudo apt-get install ros-foxy-vision-msgs
    ```
4. Install `cv_bridge` via the instructions in its [README](https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge#installation), excluding the last step (build), as it will get built later with the rest of the workspace.
5. (Optional) Most nodes with visual input are set up to run with a default USB camera. 
If you want to use it install [ROS2 USB cam](https://index.ros.org/r/usb_cam/).

Building and Running
====
1. Navigate to `~/opendr` and activate it as usual using `source bin/activate.sh`
2. Navigate to workspace root, `~/opendr/projects/opendr_ws_2` directory
3. Build the workspace:
    ```shell
    colcon build
    ```
4. Source the workspace and you are ready to go!
    ```shell
    . install/setup.bash
    ```
   Take a look at the list of tools below and click on the links to navigate to specific nodes' documentation with instructions on how to run them.

## Structure

Currently, apart from tools, opendr_ws_2 contains the following ROS2 nodes (categorized according to the input they receive):

### [Perception](src/opendr_perception/README.md)
## RGB input
1. [Pose Estimation](src/opendr_perception/README.md#pose-estimation-ros2-node)
2. [Fall Detection](src/opendr_perception/README.md#fall-detection-ros2-node)
3. [Face Detection](src/opendr_perception/README.md#face-detection-ros2-node)
4. [Face Recognition](src/opendr_perception/README.md#face-recognition-ros2-node)
5. [2D Object Detection](src/opendr_perception/README.md#2d-object-detection-ros2-nodes)
6. [2D Object Tracking - Deep Sort](src/opendr_perception/README.md#2d-object-tracking-deep-sort-ros2-node)
7. [Panoptic Segmentation](src/opendr_perception/README.md#panoptic-segmentation-ros2-node)
8. [Semantic Segmentation](src/opendr_perception/README.md#semantic-segmentation-ros2-node)
9. [Landmark-based Facial Expression Recognition](src/opendr_perception/README.md#landmark-based-facial-expression-recognition-ros2-node)
10. [Skeleton-based Human Action Recognition](src/opendr_perception/README.md#skeleton-based-human-action-recognition-ros2-node)
11. [Video Human Activity Recognition](src/opendr_perception/README.md#video-human-activity-recognition-ros2-node)
## RGB + Infrared input
1. [End-to-End Multi-Modal Object Detection (GEM)](src/opendr_perception/README.md#gem-ros2-node)
## RGBD input
1. [RGBD Hand Gesture Recognition](src/opendr_perception/README.md#rgbd-hand-gesture-recognition-ros2-node)
## Point cloud input
1. [3D Object Detection Voxel](src/opendr_perception/README.md#3d-object-detection-voxel-ros2-node)
2. [3D Object Tracking AB3DMOT](src/opendr_perception/README.md#3d-object-tracking-ab3dmot-ros2-node)
3. [2D Object Tracking FairMOT](src/opendr_perception/README.md#2d-object-tracking-fairmot-ros2-node)
## Biosignal input
1. [Heart Anomaly Detection](src/opendr_perception/README.md#heart-anomaly-detection-ros2-node)
## Audio input
1. [Speech Command Recognition](src/opendr_perception/README.md#speech-command-recognition-ros2-node)
