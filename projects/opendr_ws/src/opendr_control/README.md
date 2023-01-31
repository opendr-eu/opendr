# Control Package

This package contains ROS nodes related to control package of OpenDR.

## Setup

### Robot ROS Node
You fisrt need to start your robot nodes including its MoveIt configuration. 

If you are using a Panda Arm you can run:
```shell
roslaunch panda_moveit_config franka_control.launch robot_ip:=<robot-ip>
```

If you need to configure your robot for use with MoveIt you can use the [MoveIt Setup Assistant](https://ros-planning.github.io/moveit_tutorials/doc/setup_assistant/setup_assistant_tutorial.html).

### Camera calibration

You need to make sure that the robot tf tree and the camera tf tree are connected to be able to get the detected grasp poses in the appropriate coordinates frame. 

To do that you can, for example, use the [MoveIt Calibration package](https://ros-planning.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html).

### Camera ROS Node

Instructions to start detecting grasp poses can be found in the [perception package documentation](../perception/README.md#grasp-pose-detection-ros-node).

The `opendr_grasp_pose_detection node` needs `--only_visualize` to be set to `False` and `camera_tf_frame`, `robot_tf_frame`, `ee_tf_frame` should be set according to the camera and robot being used.


## Robot control ROS Node

To launch the robot control nodes, execute the following commands:
```shell
roslaunch control robot_control.launch 
```
The following optional arguments are available:
- -`--group_name`:  group name to be used to instantiate a MoveGroupCommander object (default=`panda_arm`)
- `--arm `: indicate which controller to use to actuate the arm (default=`opendr`)
- `--gripper`: indicate which controller to use to actuate the gripper (default=`opendr`)

Assuming the vision and robot nodes have been launched successfully you can now examine the detected grasp pose with respect to the coordinates frame `robot_tf_frame` runnning `rostopic echo /opendr/grasp_detected`.

### Arm Controller Node

This node exposes the following services: 
- `/opendr/rotate_ee (theta)`: rotates the end effector to angle `theta` (in radian)
- `/opendr/stop_action ()`: pauses the arm.
- `/opendr/resume_action ()`: resumes an action paused using the `/opendr/stop_action` service.
- `/opendr/set_joint_state ([q])`: sends the arm to a joint configuration `q`, one value per joint.
- `/opendr/set_pose_target (Pose)`: sends the end effector to a `geometry_msgs/Pose`.
- `/opendr/set_pose_target_2D ([x,y])`: sends the effector to a point `[x,y]`.
- `/opendr/set_pose_target_1D (z)`: moves the effector vertically to a height `z`.

### Gripper Node

This node exposes the following services:
- `/opendr/move_gripper (width, speed)`: moves to a target `width` with the defined `speed`.
- `/opendr/grasp (width, force)`: tries to grasp at the desired `width` with a desired `force`.

### Pick And Place node

This node instantiates and starts the following action servers:
- `/opendr/pick (width, force, Pose)`: tries to pick an object of a certain `width` at a `Pose` with `force`. It approaches the object by moving above it and then slowly going down until being in a position to grasp the object. Once the object is grasped the arm retreats by moving vertically, lifting the object. The action definition is the following:

```yaml
#goal definition
float32 width
float32 force
geometry_msgs/Pose pose
---
#result definition
bool success
---
#feedback
```
- `/opendr/place (Pose)`: tries to place an object at `Pose`. In a similar way as the `pick` action, it approaches the goal pose from above, then slowly moves down and opens the gripper after reaching the desired pose. The action definition is the following:

```yaml
#goal definition
geometry_msgs/Pose pose
---
#result definition
bool success
---
#feedback
```

> **_NOTE:_**  Remember to take into account the height of the object when setting goal poses.

## Pick and Place example 

This example instantiates a `PickAndPlaceClient` and uses the information from the grasp detection topics to set the pick and place goal poses. 
To launch the example, execute the following command:
```shell
roslaunch control pick_and_place_example.launch 
```
The script instantiates a `Detection` class to keep track of the results sent by the peception node. This class contains a dictionary indexed with the class id of the detected objects and for each stores their pose. To avoid storing the same object multiple time it calculates the distance between the new detection and the ones already stored for the same class id. It also contains the two following functions:
- To retrieve the class id using the object's name  
```python
def find_object_by_category(self, category_name):
  '''
  Use the object's name to find its class id.
  :param category_name: name of the object
  :type category_name: str
  :return: class id
  :rtype: int
  '''
```
This uses the information contained in the ROS parameter server at a location given by the VisionInfo message in `/opendr/object_categories`.

- To retrieve the object's pose using a class id

```python
def get_object_pose(self, object_id):
  '''
  Use the class id to find the pose of one of the corresponding objects.
  :param object_id: class id of the objects to look fo
  :type category_name: int
  :return: object pose
  :rtype: geometry_msgs/Pose
  '''
```
