# Perception Package

This package contains ROS nodes related to perception package of OpenDR.

## Pose Estimation ROS Node
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH="/home/user/opendr/src:$PYTHONPATH"
```

2. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node
```

3. You are then ready to start the pose detection node

```shell
rosrun perception pose_estimation.py
```

4. You can examine the annotated image stream using `rqt_image_view` (select the topic `/opendr/image_pose_annotated`) or
   `rostopic echo /opendr/poses`

## Face Recognition ROS Node
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH="/home/user/opendr/src:$PYTHONPATH"
```

2. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node
```

3. You are then ready to start the face recognition node

```shell
rosrun perception face_recognition.py
```

4. The database entry and the returned confidence is published under the topic name `/opendr/face_recognition`, and the human-readable ID
under `/opendr/face_recognition_id`.

## 2D Object Detection ROS Nodes
ROS nodes are implemented for the SSD, YOLOv3, CenterNet and DETR generic object detectors. Steps 1, 2 from above must run first.
Then, to initiate the SSD detector node, run:

```shell
rosrun perception object_detection_2d_ssd.py
```
The annotated image stream can be viewed using `rqt_image_view`, and the default topic name is
`/opendr/image_boxes_annotated`. The bounding boxes alone are also published as `/opendr/objects`.
Similarly, the YOLOv3, CenterNet and DETR detector nodes can be run with:
```shell
rosrun perception object_detection_2d_yolov3.py
```
or
```shell
rosrun perception object_detection_2d_centernet.py
```
or
```shell
rosrun perception object_detection_2d_detr.py
```
respectively.

## Face Detection ROS Node
A ROS node for the RetinaFace detector is implemented, supporting both the ResNet and MobileNet versions, the latter of
which performs mask recognition as well. After setting up the environment, the detector node can be initiated as:
```shell
rosrun perception face_detection_retinaface.py
```
The annotated image stream is published under the topic name `/opendr/image_boxes_annotated`, and the bounding boxes alone
under `/opendr/faces`.

## GEM ROS Node
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH="/home/user/opendr/src:$PYTHONPATH"
```
2. First one needs to find points in the color and infrared images that correspond, in order to find the homography matrix that allows to correct for the difference in perspective between the infrared and the RGB camera.
These points can be selected using a [utility tool](../../../../src/opendr/perception/object_detection_2d/utils/get_color_infra_alignment.py) that is provided in the toolkit.

3. Pass the points you have found as *pts_color* and *pts_infra* arguments to the ROS gem.py node.

4. Start the node responsible for publishing images. If you have a RealSense camera, then you can use the corresponding node (assuming you have installed [realsense2_camera](http://wiki.ros.org/realsense2_camera)):

```shell
roslaunch realsense2_camera rs_camera.launch enable_color:=true enable_infra:=true enable_depth:=false enable_sync:=true infra_width:=640 infra_height:=480
```

4. You are then ready to start the pose detection node

```shell
rosrun perception object_detection_2d_gem.py
```

5. You can examine the annotated image stream using `rqt_image_view` (select one of the topics `/opendr/color_detection_annotated` or `/opendr/infra_detection_annotated`) or `rostopic echo /opendr/detections`


## Panoptic Segmentation ROS Node
A ROS node for performing panoptic segmentation on a specified RGB image stream using the [EfficientPS](../../../../src/opendr/perception/panoptic_segmentation/README.md) network.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception panoptic_segmentation_efficient_ps.py CHECKPOINT IMAGE_TOPIC
```
with `CHECKPOINT` pointing to the path to the trained model weights and `IMAGE_TOPIC` specifying the ROS topic, to which the node will subscribe.

Additionally, the following optional arguments are available:
- `-h, --help`: show a help message and exit
- `--heamap_topic HEATMAP_TOPIC`: publish the semantic and instance maps on `HEATMAP_TOPIC`
- `--visualization_topic VISUALIZATION_TOPIC`: publish the panoptic segmentation map as an RGB image on `VISUALIZATION_TOPIC` or a more detailed overview if using the `--detailed_visualization` flag
- `--detailed_visualization`: generate a combined overview of the input RGB image and the semantic, instance, and panoptic segmentation maps


## Semantic Segmentation ROS Node
A ROS node for performing semantic segmentation on an input image using the BiseNet model.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception semantic_segmentation_bisenet.py IMAGE_TOPIC
```

Additionally, the following optional arguments are available:
- `-h, --help`: show a help message and exit
- `--heamap_topic HEATMAP_TOPIC`: publish the heatmap on `HEATMAP_TOPIC`

## RGBD Hand Gesture Recognition ROS Node

A ROS node for performing hand gesture recognition using MobileNetv2 model trained on HANDS dataset. The node has been tested with Kinectv2 for depth data acquisition with the following drivers: https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai-iai_kinect2. Assuming that the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception rgbd_hand_gesture_recognition.py
```
The predictied classes are published to the topic `/opendr/gestures`.

