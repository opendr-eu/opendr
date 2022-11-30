# Control Package

This package contains ROS nodes related to control package of OpenDR.

## Setup

### Robot ROS Node
You fisrt need to start your robot nodes including its MoveIt configuration. 

If you are using a Panda Arm you can run:
```shell
roslaunch panda_moveit_config franka_control.launch robot_ip:=<robot_ip>
```

If you need to configure your robot for use with MoveIt you can use the [MoveIt Setup Assistant](https://ros-planning.github.io/moveit_tutorials/doc/setup_assistant/setup_assistant_tutorial.html).

### Camera calibration

You need to make sure that the robot tf tree and the camera tf tree are connected to be able to get the detected grasp poses in the appropriate coordinates frame. 

To do that you can, for example, use the [MoveIt Calibration package](https://ros-planning.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html).

### Camera ROS Node

Instructions to start detecting grasp poses can be found in the [perception package documentation](../perception/README.md#grasp-pose-detection-ros-node).

The `opendr_grasp_pose_detection node` needs `--only_visualize` to be set to `False` and `camera_tf_frame`, `robot_tf_frame`, `ee_tf_frame` should be set according to the camera and robot being used.


## Robot control ROS Node