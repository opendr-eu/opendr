## opendr_bridge package


This *opendr_bridge* package provides an interface to convert OpenDR data types and targets into ROS-compatible ones similar to CvBridge.
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

- **message**: *sensor_msgs.msg.Img*\
  ROS image to be converted into an OpenDR image.
- **encoding**: *str, default='bgr8'*\
  Encoding to be used for the conversion (inherited from CvBridge).

#### `ROSBridge.to_ros_image`

```python
ROSBridge.to_ros_image(self,
                       image,
                       encoding)
```

This method converts an OpenDR image into a ROS image.

Parameters:

- **message**: *engine.data.Image*\
  OpenDR image to be converted into a ROS message.
- **encoding**: *str, default='bgr8'*\
  Encoding to be used for the conversion (inherited from CvBridge).

#### `ROSBridge.from_ros_pose`

```python
ROSBridge.from_ros_pose(self,
                        ros_pose)
```

Converts an OpenDRPose2D message into an OpenDR Pose.

Parameters:

- **ros_pose**: *opendr_bridge.msg.OpenDRPose2D*\
  ROS pose to be converted into an OpenDR Pose.

#### `ROSBridge.to_ros_pose`

```python
ROSBridge.to_ros_pose(self,
                      pose)
```
Converts an OpenDR Pose into a OpenDRPose2D msg that can carry the same information, i.e. a list of keypoints,
the pose detection confidence and the pose id.
Each keypoint is represented as an OpenDRPose2DKeypoint with x, y pixel position on input image with (0, 0)
being the top-left corner.

Parameters:

- **pose**: *engine.target.Pose*\
  OpenDR Pose to be converted to ROS OpenDRPose2D.


#### `ROSBridge.to_ros_category`

```python
ROSBridge.to_ros_category(self,
                          category)
```
Converts an OpenDR Category used for category recognition into a ROS ObjectHypothesis.

Parameters:

- **message**: *engine.target.Category*\
  OpenDR Category used for category recognition to be converted to ROS ObjectHypothesis.

#### `ROSBridge.to_ros_category_description`

```python
ROSBridge.to_ros_category_description(self,
                                      category)
```
Converts an OpenDR Category into a ROS String.

Parameters:

- **message**: *engine.target.Category*\
  OpenDR Category to be converted to ROS String.


#### `ROSBridge.from_ros_category`

```python
ROSBridge.from_ros_category(self,
                            ros_hypothesis)
```

Converts a ROS ObjectHypothesis message into an OpenDR Category.

Parameters:

- **message**: *vision_msgs.msg.ObjectHypothesis*\
  ROS ObjectHypothesis to be converted into an OpenDR Category.


#### `ROSBridge.from_ros_face`

```python
ROSBridge.from_ros_face(self,
                        ros_hypothesis)
```

Converts a ROS ObjectHypothesis message into an OpenDR Category.

Parameters:

- **message**: *vision_msgs.msg.ObjectHypothesis*\
  ROS ObjectHypothesis to be converted into an OpenDR Category.

#### `ROSBridge.to_ros_face`

```python
ROSBridge.to_ros_face(self,
                      category)
```
Converts an OpenDR Category used for face recognition into a ROS ObjectHypothesis.

Parameters:

- **message**: *engine.target.Category*\
  OpenDR Category used for face recognition to be converted to ROS ObjectHypothesis.

#### `ROSBridge.to_ros_face_id`

```python
ROSBridge.to_ros_face_id(self,
                         category)
```
Converts an OpenDR Category into a ROS String.

Parameters:

- **message**: *engine.target.Category*\
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

- **ros_pose**: *geometry_msgs.msg.Pose*\
  ROS pose to be converted into an OpenDR pose.

#### `ROSBridge.to_ros_3Dpose`

```python
ROSBridge.to_ros_3Dpose(self,
                      opendr_pose)
```
Converts an OpenDR pose into a ROS ```geometry_msgs.msg.Pose``` message.

Parameters:

- **opendr_pose**: *engine.target.Pose*\
  OpenDR pose to be converted to ```geometry_msgs.msg.Pose``` message.

#### `ROSBridge.to_ros_mesh`

```python
ROSBridge.to_ros_mesh(self,
                      vertices, faces)
```
Converts a triangle mesh consisting of vertices, faces into a ROS ```shape_msgs.msg.Mesh``` message.

Parameters:

- **vertices**: *numpy.ndarray*\
  Vertices (Nx3) of a triangle mesh.
- **faces**: *numpy.ndarray*\
  Faces (Nx3) of a triangle mesh.

  #### `ROSBridge.to_ros_colors`

```python
ROSBridge.to_ros_colors(self,
                        colors)
```
Converts a list of colors into a list of ROS ```std_msgs.msg.colorRGBA``` messages.

Parameters:

- **colors**: *list of list of size 3*\
  List of colors to be converted to a list of ROS colors.

  #### `ROSBridge.from_ros_mesh`

```python
ROSBridge.from_ros_mesh(self,
                        ros_mesh)
```
Converts a ROS mesh into arrays of vertices and faces of a triangle mesh.

Parameters:

- **ros_mesh**: *shape_msgs.msg.Mesh*\

  #### `ROSBridge.from_ros_colors`

```python
ROSBridge.from_ros_colors(self,
                          ros_colors)
```
Converts a list of ROS colors into an array (Nx3).

Parameters:

- **ros_colors**: list of *std_msgs.msg.colorRGBA*


#### `ROSBridge.from_ros_image_to_depth`

```python
ROSBridge.from_ros_image_to_depth(self,
                                  message,
                                  encoding)
```

This method converts a ROS image message into an OpenDR grayscale depth image.

Parameters:

- **message**: *sensor_msgs.msg.Img*\
  ROS image to be converted into an OpenDR image.
- **encoding**: *str, default='mono16'*\
  Encoding to be used for the conversion.

#### `ROSBridge.from_category_to_rosclass`

```python
ROSBridge.from_category_to_rosclass(self,
                                    prediction,
                                    source_data)
```
This method converts an OpenDR Category object into Classification2D message with class label, confidence, timestamp and optionally corresponding input.

Parameters:

- **prediction**: *engine.target.Category*\
  OpenDR Category object
- **source_data**: *default=None*\
  Corresponding input, default=None

#### `ROSBridge.from_rosarray_to_timeseries`

```python
ROSBridge.from_rosarray_to_timeseries(self,
                                      ros_array,
                                      dim1,
                                      dim2)
```
This method converts a ROS array into OpenDR Timeseries object.

Parameters:

- **ros_array**: *std_msgs.msg.Float32MultiArray*\
  ROS array of data
- **dim1**: *int*\
  First dimension
- **dim2**: *int*\
  Second dimension

#### `ROSBridge.from_ros_point_cloud`

```python
ROSBridge.from_ros_point_cloud(self, point_cloud)
```

Converts a ROS PointCloud message into an OpenDR PointCloud.

Parameters:

- **point_cloud**: *sensor_msgs.msg.PointCloud*\
  ROS PointCloud to be converted.

#### `ROSBridge.to_ros_point_cloud`

```python
ROSBridge.to_ros_point_cloud(self, point_cloud)
```
Converts an OpenDR PointCloud message into a ROS PointCloud.

Parameters:

- **point_cloud**: *engine.data.PointCloud*\
  OpenDR PointCloud to be converted.

#### `ROSBridge.from_ros_point_cloud2`

```python
ROSBridge.from_ros_point_cloud2(self, point_cloud)
```

Converts a ROS PointCloud2 message into an OpenDR PointCloud.

Parameters:

- **point_cloud**: *sensor_msgs.msg.PointCloud2*\
  ROS PointCloud2 to be converted.

#### `ROSBridge.to_ros_point_cloud2`

```python
ROSBridge.to_ros_point_cloud2(self, point_cloud, channels)
```
Converts an OpenDR PointCloud message into a ROS PointCloud2.

Parameters:

- **point_cloud**: *engine.data.PointCloud*\
  OpenDR PointCloud to be converted.
- **channels**: *str*\
  Channels to be included in the PointCloud2 message.  
  Available channels names are ["rgb", "rgba"]

#### `ROSBridge.from_ros_boxes_3d`

```python
ROSBridge.from_ros_boxes_3d(self, ros_boxes_3d, classes)
```

Converts a ROS Detection3DArray message into an OpenDR BoundingBox3D object.

Parameters:

- **ros_boxes_3d**: *vision_msgs.msg.Detection3DArray*\
  The ROS boxes to be converted.

- **classes**: *[str]*\
  The array of classes to transform an index into a string name.

#### `ROSBridge.to_ros_boxes_3d`

```python
ROSBridge.to_ros_boxes_3d(self, boxes_3d, classes)
```
Converts an OpenDR BoundingBox3DList object into a ROS Detection3DArray message.

Parameters:

- **boxes_3d**: *engine.target.BoundingBox3DList*\
  The ROS boxes to be converted.

- **classes**: *[str]*
  The array of classes to transform from string name into an index.

#### `ROSBridge.from_ros_tracking_annotation`

```python
ROSBridge.from_ros_tracking_annotation(self, ros_detections, ros_tracking_ids, frame)
```

Converts a pair of ROS messages with bounding boxes and tracking ids into an OpenDR TrackingAnnotationList.

Parameters:

- **ros_detections**: *sensor_msgs.msg.Detection2DArray*\
  The boxes to be converted.
- **ros_tracking_ids**: *std_msgs.msg.Int32MultiArray*\
  The tracking ids corresponding to the boxes.
- **frame**: *int, default=-1*\
  The frame index to assign to the tracking boxes.

#### `ROSBridge.to_ros_single_tracking_annotation`

```python
ROSBridge.to_ros_single_tracking_annotation(self, tracking_annotation)
```

Converts a `TrackingAnnotation` object to a `Detection2D` ROS message.
This method is intended for single object tracking methods.

Parameters:

- **tracking_annotation**: *opendr.engine.target.TrackingAnnotation*\
  The box to be converted.

#### `ROSBridge.from_ros_single_tracking_annotation`

```python
ROSBridge.from_ros_single_tracking_annotation(self, ros_detection_box)
```

Converts a `Detection2D` ROS message object to a `TrackingAnnotation` object.
This method is intended for single object tracking methods.

Parameters:

- **ros_detection_box**: *vision_msgs.Detection2D*\
  The box to be converted.

## ROS message equivalence with OpenDR
1. `sensor_msgs.msg.Img` is used as an equivalent to `engine.data.Image`
2. `opendr_bridge.msg.Pose` is used as an equivalent to `engine.target.Pose`
3. `vision_msgs.msg.Detection2DArray` is used as an equivalent to `engine.target.BoundingBoxList`
4. `vision_msgs.msg.Detection2D` is used as an equivalent to `engine.target.BoundingBox` and
   to `engine.target.TrackingAnnotation` in single object tracking
5. `geometry_msgs.msg.Pose`  is used as an equivalent to `engine.target.Pose` for 3D poses conversion only.
6. `vision_msgs.msg.Detection3DArray`  is used as an equivalent to `engine.target.BoundingBox3DList`.
7. `sensor_msgs.msg.PointCloud`  is used as an equivalent to `engine.data.PointCloud`.
8. `sensor_msgs.msg.PointCloud2`  is used as an equivalent to `engine.data.PointCloud`.

## ROS services
The following ROS services are implemented (`srv` folder):
1. `opendr_bridge.OpenDRSingleObjectTracking`: can be used to initialize the tracking process of single
   object trackers, by providing a `Detection2D` bounding box