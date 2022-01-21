## engine.target Module

The [*engine.target*](/src/opendr/engine/target.py) module contains classes representing different types of targets.

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

`Target` provides the following fields that can be used by classes that inhert `Target`:
- *data*, which holds the actual predictions/annotations,
- *confidence*, which holds the prediction/annotation confidence,
- *action*, which provides the expected active perception output.


### class engine.target.Category
Bases: `engine.target.Target`

The Category target is used for 1-of-K classification problems.
It contains the predicted class or ground truth and optionally the description of the predicted class
and the prediction confidence.

The [Category](/src/opendr/engine/target.py#L105) class has the following public methods and attributes:
#### Category(prediction, confidence=None)
Construct a new *Category*.
- *prediction* is a class integer.
- *description* is an optional string describing the predicted class.
- *confidence* is an optional one-dimensional array / tensor of class probabilitiess.


### class engine.target.Keypoint
Bases: `engine.target.Target`

This target is used for keypoint detection in pose estimation, body part detection, etc.
A keypoint is a list with two coordinates [x, y], which gives the x, y position of the
keypoints on the image.

The [Keypoint](/src/opendr/engine/target.py#L192) class has the following public methods:
#### Keypoint(keypoint, confidence=None)
  Construct a new *Keypoint* object based from *keypoint*.
  *keypoint* is expected to be a list with two coordinates [x, y].


### class engine.target.Pose
Bases: `engine.target.Target`

This target is used for pose estimation. It contains a list of Keypoints.
Refer to kpt_names for keypoint naming.

The [Pose](/src/opendr/engine/target.py#L208) class has the following public methods:
#### Pose(keypoints, confidence)
  Construct a new *Pose* object based on *keypoints*.
  *keypoints* is expected to be a list of *Keypoint* objects.
  Keypoints can be accessed either by using their numerical id (e.g., pose[0]) or their name (e.g., pose['neck']). 
  Please refer to `Pose.kpt_names` for a list of supported keypoints.


### class engine.target.BoundingBox3D
Bases: `engine.target.Target`


This target is used for 3D Object Detection and describes 3D bounding box in space containing an object of interest.
Additionaly, projection of the bounding box onto camera image plane can be provided in *bbox2d* parameter.
A bounding box is described by its location (x, y, z), dimensions (w, h, d) and rotation (along vertical y axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [BoundingBox3D](/src/opendr/engine/target.py#L594) class has the following public methods:
#### BoundingBox3D(name, truncated, occluded, alpha, bbox2d, dimensions, location, rotation_y, score=0)
  Construct a new *BoundingBox3D* based on the given data.
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
  Return the `name` value of the bounding box.
#### truncated()
  Return the `truncated` value of the bounding box.
#### occluded()
  Return the `occluded` value of the bounding box.
#### alpha()
  Return the `alpha` value of the bounding box.
#### bbox2d()
  Return the `bbox2d` value of the bounding box.
#### dimensions()
  Return the `dimensions` value of the bounding box.
#### location()
  Return the `location` value of the bounding box.
#### rotation_y()
  Return the `rotation_y` value of the bounding box.

### class engine.target.BoundingBox3DList
Bases: `engine.target.Target`


This target is used for 3D object detection.
It contains a list of *BoundingBox3D*  targets that belong to the same frame.
A bounding box is described by its location (x, y, z), dimensions (l, h, w) and rotation (along vertical (y) axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [BoundingBox3DList](/src/opendr/engine/target.py#L404) class has the following public methods:
#### BoundingBox3DList(bounding_boxes_3d)
  Construct a new *BoundingBox3DList* object based on the *bounding_boxes_3d*.
  *bounding_boxes_3d* is expected to be a list of *BoundingBox3D*.
#### kitti()
  Return the annotation in KITTI format.
#### boxes()
  Return the list of *BoundingBox3D* boxes.
#### from_kitti(boxes_kitti)
  Static method that constructs *BoundingBox3DList* from the `boxes_kitti` object with KITTI annotation.


### class engine.target.TrackingAnnotation3D
Bases: `engine.target.BoundingBox3D`

This target is used for 3D object tracking and describes 3D bounding box and its unique id (accross one video or image sequence) with a frame number.
A tracking bounding box is described by frame, id, its location (x, y, z), dimensions (w, h, d) and rotation (along vertical y axis).
Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
observation angle of an object (alpha).

The [TrackingAnnotation3D](/src/opendr/engine/target.py#L472) class has the following public methods:
#### TrackingAnnotation3D(name, truncated, occluded, alpha, bbox2d, dimensions, location, rotation_y, id, score=0, frame=-1)
  Construct a new *TrackingAnnotation3D* object based on the given data.
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
  - If *with_tracking_info* is `True`, `frame` and `id` data are also returned. 
#### bounding_box_3d()
  Return the *BoundingBox3D* object constructed from this object by discarding `frame` and `id` data.
#### frame()
  Return the `frame` value of the bounding box.
#### id()
  Return the `id` value of the bounding box.


### class engine.target.TrackingAnnotation3DList
Bases: `engine.target.Target`

This target is used for 3D object detection and tracking.
It contains a list of *TrackingAnnotation3D* targets that belong to the same frame.

The [TrackingAnnotation3DList](/src/opendr/engine/target.py#L545) class has the following public methods:
#### TrackingAnnotation3DList(tracking_bounding_boxes_3d)
  Construct a new *TrackingAnnotation3DList* object based on the *tracking_bounding_boxes_3d*.
  *tracking_bounding_boxes_3d* is expected to be a list of *TrackingAnnotation3D*.
#### kitti(with_tracking_info=True)
  Return the annotation in KITTI format.
  - If *with_tracking_info* is `True`, `frame` and `id` data are also returned. 
#### boxes()
  Return the list of *TrackingAnnotation3D* boxes.
#### bounding_box_3d_list()
  Return the *BoundingBox3DList* object constructed from this object by discarding `frame` and `id` data.
#### from_kitti(boxes_kitti, ids, frames=None)
  Static method that constructs *TrackingAnnotation3DList* from the *boxes_kitti* object with KITTI annotation and corresponding *ids*.
  If *frames* is `None`, `frame` value for each object will be set to `-1`, otherwise, `frames` values are assigned.

### class engine.target.BoundingBox
Bases: `engine.target.Target`

This target is used for 2D Object Detection and describes 2D bounding box in image plane containing an object of interest.
A bounding box is described by the left-top corner and its width and height.

The [BoundingBox](/src/opendr/engine/target.py#L300) class has the following public methods:
#### BoundingBox(name, left, top, width, height, score=0)
  Construct a new *BoundingBox* object based on the given data.
  - *name* is expected to be a string or a number representing the class of the object.
  - *left* is expected to be a number representing the x position of the left-top corner.
  - *top* is expected to be a number representing the y position of the left-top corner.
  - *width* is expected to be a number representing the width of the box.
  - *height* is expected to be a number representing the height of the box.
  - *score* is expected to be a number describing the prediction confidence.
#### mot(with_confidence=True, frame=-1))
  Return the annotation in [MOT](https://motchallenge.net/instructions) format.
#### coco()
  Return the annotation in [COCO detection](https://cocodataset.org/#detection-2019) format.
  For more information and a detailed description of COCO annotations, see this [COCO annotations tutorial](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).

### class engine.target.CocoBoundingBox
Bases: `engine.target.BoundingBox`

This target is used for 2D Object Detection and describes 2D bounding box in image plane containing an object of interest in [COCO detection](https://cocodataset.org/#detection-2019) format.
A bounding box is described by the left-top corner and its width and height.
The main difference with the `BoundingBox` target, is that the `CocoBoundingBox` target can also contain information on the object's segmentation.

The [CocoBoundingBox](/src/opendr/engine/target.py#L358) class has the following public methods:
#### CocoBoundingBox(name, left, top, width, height, segmentation=[], area=0, iscrowd=0, score=0)
  Construct a new *CocoBoundingBox* object based on the given data.
  - *name* is expected to be a string or a number representing the class of the object.
  - *left* is expected to be a number representing the x position of the left-top corner.
  - *top* is expected to be a number representing the y position of the left-top corner.
  - *width* is expected to be a number representing the width of the box.
  - *height* is expected to be a number representing the height of the box.
  - *segmentation* is expected to be a list of polygon vertices around the object (`iscrowd=False`) or a run-length-encoded (RLE) bit mask (`iscrowd=True`).
  - *area* is expected to be an integer describing the area of the segmentation.
  - *iscrowd* is expected to be a bool describing whether the `CocoBoundingBox` represents a crowd (a group of objects).
    If *True*, `segmentation` is a run-length-encoded (RLE) bit mask. If *False*, `segmentation` is a list of polygon vertices around the object.
    For more information see this [tutorial](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).
  - *score* is expected to be a number describing the prediction confidence.
#### coco()
  Return the annotation in [COCO detection](https://cocodataset.org/#detection-2019) format.
  For more information and a detailed description of COCO annotations, see this [COCO annotations tutorial](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).
  If the `CocoBoundingBox` does not contain a segmentation of the object (`segmentation=[]`), the returned COCO annotation will contain no `segmentation` and
  `iscrowd` items and an `area` item with a value that is equal to the `width` times the `height` of the `CocoBoundingBox`.

### class engine.target.BoundingBoxList
Bases: `engine.target.Target`

This target is used for 2D Object Detection.
It contains a list of *BoundingBox* targets that belong to the same frame.
A bounding box is described by the left-top corner and its width and height.

The [BoundingBoxList](/src/opendr/engine/target.py#L404) class has the following public methods:
#### BoundingBoxList(name, boxes)
  Construct a new *BoundingBoxList* object based on the given data.
  - *boxes* is expected to be a list of *BoundingBox*.
#### mot(with_confidence=True)
  Return the annotation in [MOT](https://motchallenge.net/instructions) format.
#### boxes()
  Return the list of *BoundingBox* boxes.
  

### class engine.target.TrackingAnnotation
Bases: `engine.target.Target`

This target is used for 2D Object Tracking and describes 2D bounding box and its unique id (accross one video or image sequence) with a frame number.
A bounding box is described by the left-top corner and its width and height.

The [TrackingAnnotation](/src/opendr/engine/target.py#L472) class has the following public methods:
#### TrackingAnnotation(name, left, top, width, height, id, score=0, frame=-1)
  Construct a new [TrackingAnnotation](#class_engine.target.TrackingAnnotation) object based on the given data.
  - *name* is expected to be a string or a number representing the class of the object.
  - *left* is expected to be a number representing the x position of the left-top corner.
  - *top* is expected to be a number representing the y position of the left-top corner.
  - *width* is expected to be a number representing the width of the box.
  - *height* is expected to be a number representing the height of the box.
  - *id* is expected to be a number representing the object id.
  - *score* is expected to be a number describing the prediction confidence.
  - *frame* is expected to be a number describing the frame number.
#### from_mot(data)
   Static method that constructs *TrackingAnnotation* from the `data` object with MOT annotation.
#### mot(with_confidence=True)
  Return the annotation in [MOT](https://motchallenge.net/instructions) format.
#### boudning_box()
  Return the *BoundingBox* object constructed from this object.


### class engine.target.TrackingAnnotationList
Bases: `engine.target.Target`

This target is used for 2D Object Tracking.
It contains a list of *TrackingAnnotation* targets that belong to the same frame.
A bounding box is described by the left-top corner and its width and height.

The [TrackingAnnotationList](/src/opendr/engine/target.py#L545) class has the following public methods:
#### TrackingAnnotationList(name, boxes)
  Construct a new *TrackingAnnotationList* object based on the given data.
  - *boxes* is expected to be a list of *TrackingAnnotation*.
#### from_mot(data)
  Static method that constructs *TrackingAnnotationList* from the `data` object with MOT annotation.
#### mot(with_confidence=True)
  Return the annotation in [MOT](https://motchallenge.net/instructions) format.
#### bounding_box_list()
  Return the *BoundingBoxList* object constructed from this object.
#### boxes()
  Return the list of *TrackingAnnotation* boxes.


### class engine.target.Heatmap
Bases: `engine.target.Target`

This target is used for multi-class segmentation problems or multi-class problems that require heatmap annotations/outputs.

The [Heatmap](/src/opendr/engine/target.py#L994) class has the following public methods:
#### Heatmap(data, class_names=None)
  Construct a new *Heatmap* object based on the given *data*.
  - *data* is expected to be a numpy array
  - *class_names* is expected to be a dictionary and can be used for mapping class IDs (integer) to names (string) [optional]
#### numpy()
  Return a NumPy-compatible representation of *data*.
#### shape()
  Return the shape of the underlying *data* object.
#### str()
  Return a human-friendly string-based representation of *data*.
