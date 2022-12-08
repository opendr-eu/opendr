# opendr_ws

## Description
This ROS workspace contains ROS nodes and tools developed by OpenDR project. Currently, ROS nodes are compatible with ROS Noetic.
This workspace contains the `opendr_bridge` package, which provides message definitions for ROS-compatible OpenDR data types,
as well the `ROSBridge` class which provides an interface to convert OpenDR data types and targets into ROS-compatible
ones similar to CvBridge. You can find more information in the corresponding [documentation](../../docs/reference/opendr-ros-bridge.md).


## First time setup
For the initial setup you can follow the instructions below:

0. Make sure ROS noetic is installed: http://wiki.ros.org/noetic/Installation/Ubuntu (desktop full install)

1. Open a new terminal window and source the necessary distribution tools:
    ```shell
    source /opt/ros/noetic/setup.bash
    ```
   _For convenience, you can add this line to your `.bashrc` so you don't have to source the tools each time you open a  terminal window._
2. Install the following dependencies, required in order to use the OpenDR ROS tools:
    ```shell
    sudo apt-get install ros-noetic-vision-msgs ros-noetic-geometry-msgs ros-noetic-sensor-msgs ros-noetic-audio-common-msgs
    ```
3. Navigate to your OpenDR home directory (`~/opendr`) and activate the OpenDR environment using:
    ```shell
    source bin/activate.sh
    ```
    You need to do this step every time before running an OpenDR node.
4. Navigate into the OpenDR ROS workspace::
    ```shell
    cd projects/opendr_ws
    ```
5. (Optional) Most nodes with visual input are set up to run with a default USB camera. If you want to use it install the corresponding package and its dependencies:
    ```shell
    cd src
    git clone https://github.com/ros-drivers/usb_cam
    cd ..
    rosdep install --from-paths src/ --ignore-src
    ```
6. Build the packages inside the workspace:
    ```shell
    catkin_make
    ```
7. Source the workspace:
    ```shell
    source devel/setup.bash
    ```
   You are now ready to run an OpenDR ROS node, in this terminal but first the ROS master node needs to be running 

8. In a new terminal repeat step 1 and then run:
    ```shell
    roscore
    ```
   You can now return to the original terminal from step 7 and run an OpenDR ROS node. More information below.   

#### After first time setup
For running OpenDR nodes after you have completed the initial setup, you can skip steps 2 and 5 from the list above. 
You can also skip building the workspace (step 6) granted it's been already built and no changes were made to the code inside the workspace, e.g. you modified the source code of a node.

#### More information
After completing the setup you can read more information on the [opendr perception package README](src/opendr_perception/README.md), where you can find a concise list of prerequisites and helpful notes to view the output of the nodes or optimize their performance.

#### Node documentation
You can also take a look at the list of tools [below](#structure) and click on the links to navigate directly to documentation for specific nodes with instructions on how to run and modify them.

**For first time users we suggest reading the introductory sections (prerequisites and notes) first.**

## Structure

Currently, apart from tools, opendr_ws contains the following ROS nodes (categorized according to the input they receive):

### [Perception](src/opendr_perception/README.md)
## RGB input
1. [Pose Estimation](src/opendr_perception/README.md#pose-estimation-ros-node)
2. [Fall Detection](src/opendr_perception/README.md#fall-detection-ros-node)
3. [Face Detection](src/opendr_perception/README.md#face-detection-ros-node)
4. [Face Recognition](src/opendr_perception/README.md#face-recognition-ros-node)
5. [2D Object Detection](src/opendr_perception/README.md#2d-object-detection-ros-nodes)
6. [2D Object Tracking](src/opendr_perception/README.md#2d-object-tracking-ros-nodes)
7. [Panoptic Segmentation](src/opendr_perception/README.md#panoptic-segmentation-ros-node)
8. [Semantic Segmentation](src/opendr_perception/README.md#semantic-segmentation-ros-node)
9. [Landmark-based Facial Expression Recognition](src/opendr_perception/README.md#landmark-based-facial-expression-recognition-ros-node)
10. [Skeleton-based Human Action Recognition](src/opendr_perception/README.md#skeleton-based-human-action-recognition-ros-node)
11. [Video Human Activity Recognition](src/opendr_perception/README.md#video-human-activity-recognition-ros-node)
## RGB + Infrared input
1. [End-to-End Multi-Modal Object Detection (GEM)](src/opendr_perception/README.md#2d-object-detection-gem-ros-node)
## RGBD input
1. [RGBD Hand Gesture Recognition](src/opendr_perception/README.md#rgbd-hand-gesture-recognition-ros-node)
## RGB + Audio input
1. [Audiovisual Emotion Recognition](src/opendr_perception/README.md#audiovisual-emotion-recognition-ros-node)
## Audio input
1. [Speech Command Recognition](src/opendr_perception/README.md#speech-command-recognition-ros-node)
## Point cloud input
1. [3D Object Detection Voxel](src/opendr_perception/README.md#3d-object-detection-voxel-ros-node)
2. [3D Object Tracking AB3DMOT](src/opendr_perception/README.md#3d-object-tracking-ab3dmot-ros-node)
## Biosignal input
1. [Heart Anomaly Detection](src/opendr_perception/README.md#heart-anomaly-detection-ros-node)
