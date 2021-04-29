## object_tracking_3d_ab3dmot module

The *object_tracking_3d_ab3dmot* module contains the *ObjectTracking3DAb3dmotLearner* class, which inherits from the abstract class *Learner*.

### Class ObjectTracking3DAb3dmotLearner
Bases: `engine.learners.Learner`

The *ObjectTracking3DAb3dmotLearner* class is an implementation of the AB3DMOT[[1]](#object-tracking-3d-1) method.
Evaluation code is based on the KITTI evaluation development kit[[2]](#object-tracking-3d-2).
It can be used to perform 3D object tracking based on provided detections.

The [ObjectTracking3DAb3dmotLearner](#src.opendr.perception.object_tracking_3d.object_tracking_3d_ab3dmot.object_tracking_3d_ab3dmot_learner.py) class has the following public methods:

#### `ObjectTracking3DAb3dmotLearner` constructor
```python
ObjectTracking3DAb3dmotLearner(self, device, max_staleness, min_updates, state_dimensions, measurment_dimensions, state_transition_matrix, measurement_function_matrix, covariance_matrix, process_uncertainty_matrix, iou_threshold)
```

Constructor parameters:
- **device**: *{'cpu'}, default='cpu'*  
  Specifies the device to be used.
- **max_staleness**: *int, default=2*
  Specifies the maximum number of frames when no detections are associated with a track.
- **min_updates**: *int, default=3*
  Specifies the minimal number of updates for tracker to be displayed as output.
- **state_dimensions**: *int, default=10*
  Specifies the number of state dimensions for Kalman filter. Default is 10 for `x, y, z, rotation_y, w, l, h, dx, dz, drotation_y`.
- **measurement_dimensions**: *int, default=7*
  Specifies the number of measurement dimensions for Kalman filter. Default is 7 for `x, y, z, rotation_y, w, l, h`.
- **state_transition_matrix**: *numpy.ndarray, default=None*
  Specifies the [NumPy](https://numpy.org) state transition matrix for Kalman filter. If `None`, default one is used.
- **measurement_function_matrix**: *numpy.ndarray, default=None*
  Specifies the [NumPy](https://numpy.org) measurement function matrix for Kalman filter. If `None`, default one is used.
- **covariance_matrix**: *numpy.ndarray, default=None*
  Specifies the [NumPy](https://numpy.org) covariance matrix for Kalman filter. If `None`, default one is used.
- **process_uncertainty_matrix**: *numpy.ndarray, default=None*
  Specifies the [NumPy](https://numpy.org) process uncertainity matrix for Kalman filter. If `None`, default one is used.
- **iou_threshold**: *float, default=0.01*
  Specifies the minimal IoU value to match detection with a tracklet.


#### `ObjectTracking3DAb3dmotLearner.eval`
```python
ObjectTracking3DAb3dmotLearner.eval(self, dataset, logging_path, silent, verbose, count)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
- **dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `DatasetIterator`.
- **logging_path**: *str, default=None*  
  Path to save log files. If set to None, only the console will be used for logging.
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=False*  
  If set to True, enables the maximum verbosity.
- **count**: *int, default=None***  
  Specifies the number of sequences to be used for evaluation. If None, the full dataset is used.


#### `ObjectTracking3DAb3dmotLearner.infer`
```python
ObjectTracking3DAb3dmotLearner.infer(self, bounding_boxes_3d_list)
```

This method is used to perform 3D object tracking on a list of 3D bounding boxes predictions.
Returns a list of [TrackingAnnotation3DList](#class_engine.target.TrackingAnnotation3DList) objects if the list of [BoundingBox3DList](#class_engine.target.BoundingBox3DList) is given or a single [TrackingAnnotation3DList](#class_engine.target.TrackingAnnotation3DList) if a single [BoundingBox3DList](#class_engine.target.BoundingBox3DList) is given.

Parameters:
- **bounding_boxes_3d_list**: *[BoundingBox3DList](#class_engine.target.BoundingBox3DList)* or a list of *[BoundingBox3DList](#class_engine.target.BoundingBox3DList)***  
  Input data.


#### Examples

* **Inference example.**
  ```python
  import os
  from opendr.perception.object_tracking_3d.datasets.kitti_tracking import (
    KittiTrackingDatasetIterator,
  )
  from opendr.perception.object_tracking_3d.ab3dmot.object_tracking_3d_ab3dmot_learner import (
    ObjectTracking3DAb3dmotLearner,
  )

  DEVICE = "cpu"
  temp_dir = "temp"

  dataset = KittiTrackingDatasetIterator(data_path, data_path, "tracking")
  learner = ObjectTracking3DAb3dmotLearner()

  result = learner.infer(self.dataset[0][0][:5])

  print(result)

  ```

* **Evaluation example.**
  ```python
  import os
  from opendr.perception.object_tracking_3d.datasets.kitti_tracking import (
    KittiTrackingDatasetIterator,
  )
  from opendr.perception.object_tracking_3d.ab3dmot.object_tracking_3d_ab3dmot_learner import (
    ObjectTracking3DAb3dmotLearner,
  )

  DEVICE = "cpu"
  temp_dir = "temp"

  dataset = KittiTrackingDatasetIterator(data_path, data_path, "tracking")
  learner = ObjectTracking3DAb3dmotLearner()

  results = learner.eval(dataset, count=2)

  for key, val in results.items():
    print(key)
    print(val)

  ```


#### References
<a name="#object-tracking-3d-1" href="https://arxiv.org/abs/2008.08063">[1]</a> AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics,
[arXiv](https://arxiv.org/abs/2008.08063).  
<a name="#object-tracking-3d-2" href="http://www.cvlibs.net/datasets/kitti/eval_tracking.php">[2]</a> KITTI evaluation development kit.  
