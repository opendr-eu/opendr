## ROSBridge Package


This *ROSBridge* package provides an interface to convert OpenDR data types and targets into ROS-compatible ones similar to CvBridge.
The *ROSBridge* class provides two methods for each data type X:
1. *from_ros_X()* : converts the ROS equivalent of X into OpenDR data type
2. *to_ros_X()* : converts the OpenDR data type into the ROS equivalent of X

### Class ROSBridge

The *ROSBridge* class provides an interface to convert OpenDR data types and targets into ROS-compatible ones.

The ROSBridge class has the following public methods:

#### `ROSBridge` constructor
The constructor only initializes the state of the class and does not require any input arguments.
```python
ROSBridge(self)
```

#### `ROSBridge.from_ros_image`

```python
ROSBridge.from_ros_image(self,
                         message,
                         encoding)
```

This method converts a ROS Image into an OpenDR image.

Parameters:

- **message**: *sensor_msgs.msg.Img*  
  ROS image to be converted into an OpenDR image.
- **encoding**: *str, default='bgr8'*  
  Encoding to be used for the conversion (inherited from CvBridge).

#### `ROSBridge.to_ros_image`

```python
ROSBridge.to_ros_image(self,
                          image,
                          encoding)
```

This method converts an OpenDR image into a ROS image.

Parameters:

- **message**: *engine.data.Image*  
  OpenDR image to be converted into a ROS message
- **encoding**: *str, default='bgr8'*  
  Encoding to be used for the conversion (inherited from CvBridge)

#### `ROSBridge.from_ros_pose`

```python
ROSBridge.from_ros_pose(self,
                          ros_pose)
```

Converts a ROS pose into an OpenDR pose.

Parameters:

- **message**: *ros_bridge.msg.Pose*  
  ROS pose to be converted into an OpenDR pose
  
#### `ROSBridge.to_ros_pose`

```python
ROSBridge.to_ros_pose(self,
                          ros_pose)
```
Converts an OpenDR pose into a ROS pose.

Parameters:

- **message**: *engine.target.Pose*  
  OpenDR pose to be converted to ROS pose.


## ROS message equivalence with OpenDR
1. `sensor_msgs.msg.Img` is used as an equivelant to `engine.data.Image`
2. `ros_bridge.msg.Pose` is used as an equivelant to `engine.target.Pose`
