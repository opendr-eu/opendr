## engine.target Module

The *engine.target* module contains classes representing different types of targets.

### Class engine.target.BaseTarget
Bases: `abc.ABC`

Root BaseTarget class has been created to allow for setting the hierarchy of different targets.
Classes that inherit from BaseTarget can be used either as outputs of an algorithm or as ground
truth annotations, but there is no guarantee that this is always possible, i.e. that both options are possible.

Classes that are only used either for ground truth annotations or algorithm outputs must inherit this class.


### class engine.target.Target
Bases: `engine.target.BaseTarget`

Classes inheriting from the Target class always guarantee that they can be used for both cases, outputs and
ground truth annotations.
Therefore, classes that are only used to provide ground truth annotations
must inherit from BaseTarget instead of Target. To allow representing different types of
targets, this class serves as the basis for the more specialized forms of targets.
All the classes should implement the corresponding setter/getter functions to ensure that the necessary
type checking is performed (if there is no other technical obstacle to this, e.g., negative performance impact).


### class engine.target.Keypoint
Bases: `engine.target.Target`

This target is used for keypoint detection in pose estimation, body part detection, etc.
A keypoint is a list with two coordinates [x, y], which gives the x, y position of the
keypoints on the image.

The [Keypoint](#class_engine.target.Keypoint) class has the following public methods:
#### Keypoint(keypoint, confidence=None)
  Construct a new [Keypoint](#class_engine.target.Keypoint) object based from *keypoint*.
  *keypoint* is expected to be a list with two coordinates [x, y].


### class engine.target.Pose
Bases: `engine.target.Target`

This target is used for pose estimation. It contains a list of Keypoints.
Refer to kpt_names for keypoint naming.

The [Pose](#class_engine.target.Pose) class has the following public methods:
#### Pose(keypoints, confidence)
  Construct a new [Pose](#class_engine.target.Pose) object based on *keypoints*.
  *keypoints* is expected to be a list of [Keypoint](#class_engine.target.Keypoint) objects.


### class engine.target.BoundingBox3D
Bases: `engine.target.Target`

This target is used for 3D object detection and tracking.
A bounding box is described by its location (x, y, z), dimensions (w, h, d) and rotation (along vertical y axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [BoundingBox3D](#class_engine.target.BoundingBox3D) class has the following public methods:
#### BoundingBox3D(name, truncated, occluded, alpha, bbox2d, dimensions, location, rotation_y, score=0)
  Construct a new [BoundingBox3D](#class_engine.target.BoundingBox3D) object based on the given data.
  - *name* is expected to be a string with the name of detected object.
  - *truncated* is expected to be a number describing the truncation level of an object.
  - *occluded* is expected to be a number describing the occlusion level of an object.
  - *alpha* is expected to be a number describing the observation angle.
  - *bbox2d* is expected to be a list of numbers describing the 2D bounding box of an object in image plane.
  - *dimensions* is expected to be a list of numbers describing the 3D size of an object.
  - *location* is expected to be a list of numbers describing the 3D location of an object.
  - *rotation_y* is expected to be a number describing the rotation of an object along the vertical axis.
  - *score* is expected to be a number describing the prediction confidence.
#### kitti()
  Return the annotation in KITTI format.
#### name()
  Property. Returns the `name` value of the bounding box.
#### occluded()
  Property. Returns the `occluded` value of the bounding box.
#### alpha()
  Property. Returns the `alpha` value of the bounding box.
#### location()
  Property. Returns the `location` value of the bounding box.
#### rotation_y()
  Property. Returns the `rotation_y` value of the bounding box.

### class engine.target.BoundingBox3DList
Bases: `engine.target.Target`

This target is used for 3D object detection and tracking. It contains a list of BoundingBox3D targets.
A bounding box is described by its location (x, y, z), dimensions (l, h, w) and rotation (along vertical (y) axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [BoundingBox3DList](#class_engine.target.BoundingBox3DList) class has the following public methods:
#### BoundingBox3DList(bounding_boxes_3d)
  Construct a new [BoundingBox3DList](#class_engine.target.BoundingBox3DList) object based on the *bounding_boxes_3d*.
  *bounding_boxes_3d* is expected to be a list of [BoundingBox3D](#class_engine.target.BoundingBox3D).
#### kitti()
  Return the annotation in KITTI format.
#### boxes()
  Property. Returns the list of [BoundingBox3D](#class_engine.target.BoundingBox3D) boxes.
#### from_kitti(boxes_kitti)
  Static method that constructs [BoundingBox3DList](#class_engine.target.BoundingBox3DList) from the `boxes_kitti` object with KITTI annotation.


### class engine.target.TrackingBoundingBox3D
Bases: `engine.target.BoundingBox3D`

This target is used for 3D object tracking.
A tracking bounding box is described by frame, id, its location (x, y, z), dimensions (w, h, d) and rotation (along vertical y axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [TrackingBoundingBox3D](#class_engine.target.TrackingBoundingBox3D) class has the following public methods:
#### TrackingBoundingBox3D(name, truncated, occluded, alpha, bbox2d, dimensions, location, rotation_y, id, score=0, frame=-1)
  Construct a new [TrackingBoundingBox3D](#class_engine.target.TrackingBoundingBox3D) object based on the given data.
  - *name* is expected to be a string with the name of detected object.
  - *truncated* is expected to be a number describing the truncation level of an object.
  - *occluded* is expected to be a number describing the occlusion level of an object.
  - *alpha* is expected to be a number describing the observation angle.
  - *bbox2d* is expected to be a list of numbers describing the 2D bounding box of an object in image plane.
  - *dimensions* is expected to be a list of numbers describing the 3D size of an object.
  - *location* is expected to be a list of numbers describing the 3D location of an object.
  - *rotation_y* is expected to be a number describing the rotation of an object along the vertical axis.
  - *id* is a unique object id associated with the current box.
  - *score* is expected to be a number describing the prediction confidence.
  - *frame* is an index of a frame for this box.
#### kitti(with_tracking_info=True)
  Return the annotation in KITTI format.
  - If *with_tracking_info* is true, `frame` and `id` data are also returned. 
#### bounding_box_3d()
  Return the [BoundingBox3D](#class_engine.target.BoundingBox3D) object constructed from this object by discarding `frame` and `id` data.
#### name()
  Property. Returns the `name` value of the bounding box.
#### occluded()
  Property. Returns the `occluded` value of the bounding box.
#### alpha()
  Property. Returns the `alpha` value of the bounding box.
#### location()
  Property. Returns the `location` value of the bounding box.
#### rotation_y()
  Property. Returns the `rotation_y` value of the bounding box.
#### frame()
  Property. Returns the `frame` value of the bounding box.
#### id()
  Property. Returns the `id` value of the bounding box.


### class engine.target.TrackingBoundingBox3DList
Bases: `engine.target.Target`

This target is used for 3D object detection and tracking. It contains a list of TrackingBoundingBox3DList targets.
A tracking bounding box is described by frame, id, its location (x, y, z), dimensions (l, h, w) and rotation (along vertical (y) axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [TrackingBoundingBox3DList](#class_engine.target.TrackingBoundingBox3DList) class has the following public methods:
#### TrackingBoundingBox3DList(bounding_boxes_3d)
  Construct a new [TrackingBoundingBox3DList](#class_engine.target.TrackingBoundingBox3DList) object based on the *bounding_boxes_3d*.
  *bounding_boxes_3d* is expected to be a list of [TrackingBoundingBox3D](#class_engine.target.TrackingBoundingBox3D).
#### kitti(with_tracking_info=True)
  Return the annotation in KITTI format.
  - If *with_tracking_info* is true, `frame` and `id` data are also returned. 
#### boxes()
  Property. Returns the list of [TrackingBoundingBox3D](#class_engine.target.TrackingBoundingBox3D) boxes.
#### bounding_box_3d_list()
  Returns the [BoundingBox3DList](#class_engine.target.BoundingBox3DList) object cunstructed from this object by discarding `frame` and `id` data.
#### from_kitti(boxes_kitti)
  Static method that constructs [BoundingBox3DList](#class_engine.target.BoundingBox3DList) from the `boxes_kitti` object with KITTI annotation.


### class engine.target.SpeechCommand
Bases: `engine.target.Target`

This target is used for speech command recognition. Contains the predicted class or ground truth
and optionally the prediction confidence.

The [SpeechCommand](#class_engine.target.SpeechCommand) class has the following public methods:
#### SpeechCommand(prediction, confidence=None)
Construct a new [SpeechCommand](#class_engine.target.SpeechCommand) object based from *prediction*.
*prediction* is expected to be an integer designating the class and optional *confidence* a float between 0 and 1.