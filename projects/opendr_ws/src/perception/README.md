# Perception Package

This package contains ROS nodes related to perception package of OpenDR.

## Pose Estimation ROS Node
Assuming that we have already [build your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can 

1. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node 
```

2. You are then ready to start the pose detection node

```shell
rosrun perception pose_estimation.py
```

3. You can examine the annotated image stream using `rqt_image_view` or `rostopic echo /opendr/poses`



