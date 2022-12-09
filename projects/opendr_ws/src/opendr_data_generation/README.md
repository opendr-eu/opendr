# Perception Package

This package contains ROS nodes related to data generation package of OpenDR.

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

3. You are then ready to start the synthetic data generation node

```shell
rosrun data_generation synthetic_facial_generation.py
```

3. You can examine the published multiview facial images stream using `rosrun rqt_image_view rqt_image_view` (select the topic `/opendr/synthetic_facial_images`) or `rostopic echo /opendr/synthetic_facial_images`


