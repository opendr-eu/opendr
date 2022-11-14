## fall_detection module

The *fall_detection* module contains the *FallDetectorLearner* class, which inherits from the abstract class *Learner*.

### Class FallDetectorLearner
Bases: `engine.learners.Learner`

The *FallDetectorLearner* class contains the implementation of a rule-based fall detector algorithm.
It can be used to perform fall detection on images (inference) using a pose estimator.

This rule-based method can provide **cheap and fast** fall detection capabilities when pose estimation
is already being used. Its inference time cost is ~0.1% of pose estimation, adding negligible overhead.

However, it **has known limitations** due to its nature. Working with 2D poses means that depending on the 
orientation of the person, it cannot detect most fallen poses that face the camera. 
Another example of known false-positive detection occurs when a person is sitting with their knees 
detectable, but ankles obscured or undetectable, this however is critical for detecting a fallen person
whose ankles are not visible.

The [FallDetectorLearner](/src/opendr/perception/fall_detection/fall_detector_learner.py) class has the
following public methods:

#### `FallDetectorLearner` constructor
```python
FallDetectorLearner(self, pose_estimator)
```

Constructor parameters:

- **pose_estimator**: *object*\
  The provided pose estimator class used to detect poses that the fall detector uses to determine if a person has fallen.

#### `FallDetectorLearner.eval`
```python
FallDetectorLearner.eval(self, dataset, verbose)
```

This method is used to evaluate the naive fall detector algorithm on an evaluation dataset.
Returns a dictionary containing statistics regarding the evaluation.

Parameters:

- **dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **verbose**: *bool, default=True*\
  If set to True, enables the maximum verbosity.

#### `FallDetectorLearner.infer`
```python
FallDetectorLearner.infer(self, img)
```

This method is used to perform fall detection on an image.
Returns a list of tuples, one for every person detected, that each contains an `engine.target.Category`, a list of three keypoints that define two lines that are used to determine if the person has fallen, and the complete poses that were detected.
It returns an empty list if no pose detections were made.

The `engine.target.Category` is `1` if person has fallen, `-1` if person is standing and `0` if a person is detected, but
the algorithm is unable to detect if person is standing or fallen.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.

#### `FallDetectorLearner.download`
```python
FallDetectorLearner.download(path, mode, verbose, url)
```

Download utility for downloading fall detection test images and annotations.

Parameters:

- **path**: *str, default=None*\
  Local path to save the files, defaults to '.' if None.
- **mode**: *str, default="test_data"*\
  What files to download, can only be "test_data"
- **verbose**: *bool, default=False*\
  Whether to print messages in the console.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### Examples

* **Inference and result drawing example on a test image using OpenCV.**
  ```python
  import cv2
  from opendr.engine.data import Image
  from opendr.perception.fall_detection import FallDetectorLearner
  from opendr.perception.pose_estimation import LightweightOpenPoseLearner
  from opendr.perception.pose_estimation import draw, get_bbox
  
  pose_estimator = LightweightOpenPoseLearner(device="cuda", mobilenet_use_stride=False)
  pose_estimator.download(verbose=True)  # Download the default pretrained mobilenet model
  pose_estimator.load("openpose_default")
  
  fall_detector = FallDetectorLearner(pose_estimator)
  
  # Download a sample dataset
  fall_detector.download(verbose=True)
  
  img = Image.open("test_images/fallen.png")
  detections = fall_detector.infer(img)
  fallen = detections[0][0].data  # Get fallen int from first detection
  pose = detections[0][2]  # Get pose from first detection
  img = img.opencv()
  draw(img, pose)  # Draw the detected pose
  
  if fallen == 1:
      x, y, w, h = get_bbox(pose)
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
      cv2.putText(img, "Detected fallen person", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.imshow('Result', img)
  cv2.waitKey(0)
  ```
