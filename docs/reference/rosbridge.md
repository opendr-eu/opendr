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
  OpenDR image to be converted into a ROS message.
- **encoding**: *str, default='bgr8'*  
  Encoding to be used for the conversion (inherited from CvBridge).

#### `ROSBridge.from_ros_pose`

```python
ROSBridge.from_ros_pose(self,
                        ros_pose)
```

Converts a ROS pose into an OpenDR pose.

Parameters:

- **message**: *ros_bridge.msg.Pose*  
  ROS pose to be converted into an OpenDR pose.
  
#### `ROSBridge.to_ros_pose`

```python
ROSBridge.to_ros_pose(self,
                      ros_pose)
```
Converts an OpenDR pose into a ROS pose.

Parameters:

- **message**: *engine.target.Pose*  
  OpenDR pose to be converted to ROS pose.
 
#### `ROSBridge.from_ros_face`

```python
ROSBridge.from_ros_face(self,
                        ros_hypothesis)
```

Converts a ROS ObjectHypothesis message into an OpenDR Category.

Parameters:

- **message**: *ros_bridge.msg.ObjectHypothesis*  
  ROS ObjectHypothesis to be converted into an OpenDR Category.
  
#### `ROSBridge.to_ros_face`

```python
ROSBridge.to_ros_face(self,
                      category)
```
Converts an OpenDR Category used for face recognition into a ROS ObjectHypothesis.

Parameters:

- **message**: *engine.target.Category*  
  OpenDR Category used for face recognition to be converted to ROS ObjectHypothesis.
  
#### `ROSBridge.to_ros_face_id`

```python
ROSBridge.to_ros_face_id(self,
                         category)
```
Converts an OpenDR Category into a ROS String.

Parameters:

- **message**: *engine.target.Category*  
  OpenDR Category to be converted to ROS String.
  
#### `ROSBridge.to_ros_boxes`

```python
ROSBridge.to_ros_boxes(self,
                       box_list)
```
Converts an OpenDR BoundingBoxList into a Detection2DArray msg that can carry the same information. Each bounding box is 
represented by its center coordinates as well as its width/height dimensions.

#### `ROSBridge.from_ros_boxes`

```python
ROSBridge.from_ros_boxes(self,
                         ros_detections)
```
Converts a ROS Detection2DArray message with bounding boxes into an OpenDR BoundingBoxList

#### `ROSBridge.from_ros_3Dpose`

```python
ROSBridge.from_ros_3Dpose(self,
                        ros_pose)
```

Converts a ROS pose into an OpenDR pose (used for a 3D pose).

Parameters:

- **ros_pose**: *geometry_msgs.msg.Pose*  
  ROS pose to be converted into an OpenDR pose.
       
#### `ROSBridge.to_ros_3Dpose`

```python
ROSBridge.to_ros_3Dpose(self,
                      opendr_pose)
```
Converts an OpenDR pose into a ROS ```geometry_msgs.msg.Pose``` message.

Parameters:

- **opendr_pose**: *engine.target.Pose*  
  OpenDR pose to be converted to ```geometry_msgs.msg.Pose``` message.
       
#### `ROSBridge.to_ros_mesh`

```python
ROSBridge.to_ros_mesh(self,
                      vertices, faces)
```
Converts a triangle mesh consisting of vertices, faces into a ROS ```shape_msgs.msg.Mesh``` message.

Parameters:

- **vertices**: *numpy.ndarray*  
  Vertices (Nx3) of a triangle mesh.
- **faces**: *numpy.ndarray*  
  Faces (Nx3) of a triangle mesh. 
  
  #### `ROSBridge.to_ros_colors`

```python
ROSBridge.to_ros_colors(self,
                      colors)
```
Converts a list of colors into a list of ROS ```std_msgs.msg.colorRGBA``` messages.

Parameters:

- **colors**: *list of list of size 3*  
  List of colors to be converted to a list of ROS colors.
  
  #### `ROSBridge.from_ros_mesh`

```python
ROSBridge.from_ros_mesh(self,
                      ros_mesh)
```
Converts a ROS mesh into arrays of vertices and faces of a triangle mesh.

Parameters:
- **ros_mesh**: *shape_msgs.msg.Mesh* 
  
  #### `ROSBridge.from_ros_colors`

```python
ROSBridge.from_ros_colors(self,
                      ros_colors)
```
Converts a list of ROS colors into an array (Nx3).

Parameters:
- **ros_colors**: list of *std_msgs.msg.colorRGBA* 

## ROS message equivalence with OpenDR
1. `sensor_msgs.msg.Img` is used as an equivelant to `engine.data.Image`
2. `ros_bridge.msg.Pose` is used as an equivelant to `engine.target.Pose`
3. `vision_msgs.msg.Detection2DArray` is used as an equivalent to `engine.target.BoundingBoxList`
4. `vision_msgs.msg.Detection2D` is used as an equivalent to `engine.target.BoundingBox`
5. `geometry_msgs.msg.Pose`  is used as an equivelant to `engine.target.Pose` for 3D poses conversion only.
