# opendr_ws_2

## Description
This ROS2 workspace contains ROS2 nodes and tools developed by OpenDR project. Currently, ROS2 nodes are compatible with ROS2 Foxy.
This workspace contains the `opendr_ros2_bridge` package, which contains the `ROS2Bridge` class that provides an interface to convert OpenDR data types and targets into ROS-compatible
ones similar to CvBridge. The workspace also contains the `opendr_ros2_interfaces` which provides message and service definitions for ROS-compatible OpenDR data types. You can find more information in the corresponding [opendr_ros2_bridge documentation](../../docs/reference/ros2bridge.md) and [opendr_ros2_interfaces documentation](). <!-- add interfaces readme link -->

## First time setup

For the initial setup you can follow the instructions below:

0. Make sure that [ROS2-foxy is installed.](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

1. Source the necessary distribution tools:
    ```shell
    source /opt/ros/foxy/setup.bash
    ```
   _For convenience, you can add this line to your `.bashrc` so you don't have to source the tools each time you open a  terminal window._

<!--4. Install `cv_bridge` via the instructions in its [README](https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge#installation), excluding the last step (build), as it will get built later with the rest of the workspace. TODO is this needed?-->

2. Navigate to your OpenDR home directory (`~/opendr`) and activate the OpenDR environment using:
    ```shell
    source bin/activate.sh
    ```
    You need to do this step every time before running an OpenDR node.

3. Navigate into the OpenDR ROS2 workspace::
    ```shell
    cd projects/opendr_ws_2
    ```

4. Build the packages inside the workspace:
    ```shell
    colcon build
    ```

5. Source the workspace:
    ```shell
    . install/setup.bash
    ```
   You are now ready to run an OpenDR ROS node.

#### After first time setup
For running OpenDR nodes after you have completed the initial setup, you can skip steps 0 from the list above.
You can also skip building the workspace (step 4) granted it's been already built and no changes were made to the code inside the workspace, e.g. you modified the source code of a node.

#### More information
After completing the setup you can read more information on the [opendr perception package README](src/opendr_perception/README.md), where you can find a concise list of prerequisites and helpful notes to view the output of the nodes or optimize their performance.

#### Node documentation
You can also take a look at the list of tools [below](#structure) and click on the links to navigate directly to documentation for specific nodes with instructions on how to run and modify them.

**For first time users we suggest reading the introductory sections (prerequisites and notes) first.**

## Structure

Currently, apart from tools, opendr_ws_2 contains the following ROS2 nodes (categorized according to the input they receive):

### [Perception](src/opendr_perception/README.md)
## RGB input
1. [Pose Estimation](src/opendr_perception/README.md#pose-estimation-ros2-node)
2. [High Resolution Pose Estimation](src/opendr_perception/README.md#high-resolution-pose-estimation-ros2-node)
3. [Fall Detection](src/opendr_perception/README.md#fall-detection-ros2-node)
4. [Face Detection](src/opendr_perception/README.md#face-detection-ros2-node)
5. [Face Recognition](src/opendr_perception/README.md#face-recognition-ros2-node)
6. [2D Object Detection](src/opendr_perception/README.md#2d-object-detection-ros2-nodes)
7. [2D Single Object Tracking](src/opendr_perception/README.md#2d-single-object-tracking-ros2-node)
8. [2D Object Tracking](src/opendr_perception/README.md#2d-object-tracking-ros2-nodes)
9. [Panoptic Segmentation](src/opendr_perception/README.md#panoptic-segmentation-ros2-node)
10. [Semantic Segmentation](src/opendr_perception/README.md#semantic-segmentation-ros2-node)
11. [Image-based Facial Emotion Estimation](src/opendr_perception/README.md#image-based-facial-emotion-estimation-ros2-node)
12. [Landmark-based Facial Expression Recognition](src/opendr_perception/README.md#landmark-based-facial-expression-recognition-ros2-node)
13. [Skeleton-based Human Action Recognition](src/opendr_perception/README.md#skeleton-based-human-action-recognition-ros2-nodes)
14. [Video Human Activity Recognition](src/opendr_perception/README.md#video-human-activity-recognition-ros2-node)
## RGB + Infrared input
1. [End-to-End Multi-Modal Object Detection (GEM)](src/opendr_perception/README.md#2d-object-detection-gem-ros2-node)
## RGBD input
1. [RGBD Hand Gesture Recognition](src/opendr_perception/README.md#rgbd-hand-gesture-recognition-ros2-node)
## RGB + Audio input
1. [Audiovisual Emotion Recognition](src/opendr_perception/README.md#audiovisual-emotion-recognition-ros2-node)
## Audio input
1. [Speech Command Recognition](src/opendr_perception/README.md#speech-command-recognition-ros2-node)
## Point cloud input
1. [3D Object Detection Voxel](src/opendr_perception/README.md#3d-object-detection-voxel-ros2-node)
2. [3D Object Tracking AB3DMOT](src/opendr_perception/README.md#3d-object-tracking-ab3dmot-ros2-node)
## Biosignal input
1. [Heart Anomaly Detection](src/opendr_perception/README.md#heart-anomaly-detection-ros2-node)
