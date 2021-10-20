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

3. You can examine the annotated image stream using `rqt_image_view` (select the topic `/opendr/image_pose_annotated`) or `rostopic echo /opendr/poses`


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
rosrun perception gem.py
```

5. You can examine the annotated image stream using `rqt_image_view` (select one of the topics `/opendr/color_detection_annotated` or `/opendr/infra_detection_annotated`) or `rostopic echo /opendr/detections`


