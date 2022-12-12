## SiamRPNLearner module

The *SiamRPN* module contains the *SiamRPNLearner* class, which inherits from the abstract class *Learner*.

### Class SiamRPNLearner
Bases: `engine.learners.Learner`

The *SiamRPNLearner* class is a wrapper of the SiamRPN detector[[1]](#siamrpn-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo/siamrpn).
It can be used to perform object tracking on videos (inference) as well as train new object tracking models.

The [SiamRPNLearner](/src/opendr/perception/object_tracking_2d/siamrpn/siamrpn_learner.py) class has the following public methods:

#### `SiamRPNLearner` constructor
```python
SiamRPNLearner(self, device, n_epochs, num_workers, warmup_epochs, lr, weight_decay, momentum, cls_weight, loc_weight, batch_size, temp_path)
```

Parameters:

- **device**: *{'cuda', 'cpu'}, default='cuda'*\
  Specifies the device to be used.
- **n_epochs**: *int, default=50*\
  Specifies the number of epochs to be used during training.
- **num_workers**: *int, default=1*\
  Specifies the number of workers to be used when loading datasets or performing evaluation.
- **warmup_epochs**: *int, default=2*\
  Specifies the number of epochs during which the learning rate is annealed to **lr**.
- **lr**: *float, default=0.001*\
  Specifies the initial learning rate to be used during training.
- **weight_decay**: *float, default=0*\
  Specifies the weight decay to be used during training.
- **momentum**: *float, default=0.9*\
  Specifies the momentum to be used for optimizer during training.
- **cls_weight**: *float, default=1.*\
  Specifies the classification loss multiplier to be used for optimizer during training.
- **loc_weight**: *float, default=1.2*\
  Specifies the localization loss multiplier to be used for optimizer during training.
- **batch_size**: *int, default=32*\
  Specifies the batch size to be used during training.
- **temp_path**: *str, default=''*\
  Specifies a path to be used for data downloading.


#### `SiamRPNLearner.fit`
```python
SiamRPNLearner.fit(self, dataset, log_interval, n_gpus, verbose)
```

This method is used to train the algorithm on a `DetectionDataset` or `ExternalDataset` dataset and also performs evaluation on a validation set using the trained model.
Returns a dictionary containing stats regarding the training process.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
- **log_interval**: *int, default=20*\
  Training loss is printed in stdout after this amount of iterations.
- **n_gpus**: *int, default=1*\
  If CUDA is enabled, training can be performed on multiple GPUs as set by this parameter.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.

#### `SiamRPNLearner.eval`
```python
SiamRPNLearner.eval(self, dataset)
```

Performs evaluation on a dataset. The OTB dataset is currently supported.

Parameters:

- **dataset**: *object*\
  Object that holds dataset to perform evaluation on.
  Expected type is `ExternalDataset` with `otb2015` dataset type.

#### `SiamRPNLearner.infer`
```python
SiamRPNLearner.infer(self, img, init_box)
```

Performs inference on a single image.
If the `init_box` is provided, the tracker is initialized.
If not, the current position of the target is updated by running inference on the image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.
- **init_box**: *object, default=None*\
  Object of type engine.target.TrackingAnnotation.
  If provided, it is used to initialize the tracker.

#### `SiamRPNLearner.save`
```python
SiamRPNLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path.
The model name is extracted from the base folder in the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved.
  The model name is extracted from the base folder of this path.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.

#### `SiamRPNLearner.load`
```python
SiamRPNLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.

#### `SiamRPNLearner.download`
```python
SiamRPNLearner.download(self, path, mode, verbose, url, overwrite)
```

Downloads data needed for the various functions of the learner, e.g., pre-trained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded.
  If *None*, the *self.temp_path* directory is used instead.
- **mode**: *{'pretrained', 'video', 'test_data', 'otb2015'}, default='pretrained'*\
  If *'pretrained'*, downloads a pre-trained detector model.
  If *'video'*, downloads a single video to perform inference on.
  If *'test_data'* downloads a dummy version of the OTB dataset for testing purposes.
  If *'otb2015'*, attempts to download the OTB dataset (100 videos).
  This process lasts a long time.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.
- **overwrite**: *bool, default=False*\
  If True, files will be re-downloaded if they already exists.
  This can solve some issues with large downloads.

#### Examples

* **Training example using `ExternalDataset` objects**.
  Training is supported solely via the `ExternalDataset` class.
  See [class README](/src/opendr/perception/object_tracking_2d/siamrpn/README.md) for a list of supported datasets and presumed data directory structure.
  Example training on COCO Detection dataset:
  ```python
  from opendr.engine.datasets import ExternalDataset
  from opendr.perception.object_tracking_2d import SiamRPNLearner

  dataset = ExternalDataset("/path/to/data/root", "coco")
  learner = SiamRPNLearner(device="cuda", n_epochs=50, batch_size=32,
                           lr=1e-3)
  learner.fit(dataset)
  learner.save("siamrpn_custom")
  ```

* **Inference and result drawing example on a test mp4 video using OpenCV.**
  ```python
  import cv2
  from opendr.engine.target import TrackingAnnotation
  from opendr.perception.object_tracking_2d import SiamRPNLearner

  learner = SiamRPNLearner(device="cuda")
  learner.download(".", mode="pretrained")
  learner.load("siamrpn_opendr")

  learner.download(".", mode="video")
  cap = cv2.VideoCapture("tc_Skiing_ce.mp4")

  init_bbox = TrackingAnnotation(left=598, top=312, width=75, height=200, name=0, id=0)

  frame_no = 0
  while cap.isOpened():
      ok, frame = cap.read()
      if not ok:
          break

      if frame_no == 0:
          # first frame, pass init_bbox to infer function to initialize the tracker
          pred_bbox = learner.infer(frame, init_bbox)
      else:
          # after the first frame only pass the image to infer
          pred_bbox = learner.infer(frame)

      frame_no += 1

      cv2.rectangle(frame, (pred_bbox.left, pred_bbox.top),
                    (pred_bbox.left + pred_bbox.width, pred_bbox.top + pred_bbox.height),
                    (0, 255, 255), 3)
      cv2.imshow('Tracking Result', frame)
      cv2.waitKey(1)

  cv2.destroyAllWindows()
  ```


#### Performance evaluation

We have measured the performance on the OTB2015 dataset in terms of success and FPS on an RTX 2070.
```
------------------------------------------------
|       Tracker name       | Success |   FPS   |
------------------------------------------------
| siamrpn_alexnet_v2_otb15 |  0.668  |  132.1  |
------------------------------------------------
```

#### References
<a name="siamrpn-1" href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf">[1]</a>
High Performance Visual Tracking with Siamese Region Proposal Network,
[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf).
