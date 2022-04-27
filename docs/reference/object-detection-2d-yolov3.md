## YOLOv3DetectorLearner module

The *yolov3* module contains the *YOLOv3DetectorLearner* class, which inherits from the abstract class *Learner*.

### Class YOLOv3DetectorLearner
Bases: `engine.learners.Learner`

The *YOLOv3DetectorLearner* class is a wrapper of the YOLO detector[[1]](#yolo-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/yolo/yolo3.py).
It can be used to perform object detection on images (inference) as well as train new object detection models.

The [YOLOv3DetectorLearner](/src/opendr/perception/object_detection_2d/yolov3/yolov3_learner.py) class has the following
public methods:

#### `YOLOv3DetectorLearner` constructor
```python
YOLOv3DetectorLearner(self, lr, epochs, batch_size, device, backbone, img_size,
                      lr_schedule, temp_path, checkpoint_after_iter, checkpoint_load_iter,
                      val_after, log_after, num_workers, weight_decay, momentum, mixup,
                      no_mixup_epochs, label_smoothing, random_shape, warmup_epochs, lr_decay_period,
                      lr_decay_epoch, lr_decay)
```

Constructor parameters:

- **lr**: *float, default=0.0001*\
  Specifies the initial learning rate to be used during training.
- **epochs**: *int, default=100*\
  Specifies the number of epochs to be used during training.
- **batch_size**: *int, default=2*\
  Specifies the batch size to be used during training.
- **device**: *{'cuda', 'cpu'}, default='cuda'*\
  Specifies the device to be used.
- **backbone**: *{'darknet53', 'mobilenet1.0', 'mobilenet0.25'}, default='darknet53'*\
  Specifies the backbone architecture.
- **img_size**: *int, default=416*\
  Specifies the image size to be used during training. Must be a multiple of *32*.
- **lr_schedule**: *{'step', 'cosine', 'poly'}, default='step'*\
  Specifies the desired learning rate schedule.
- **temp_path**: *str, default=''*\
  Specifies a path to be used for storage of checkpoints during training.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
- **val_after**: *int, default=5*\
  Epoch interval between evaluations during training.
- **log_after**: *int, default=100*\
  Specifies interval (in iterations/batches) between information logging on *stdout*.
- **num_workers**: *int, default=8*\
  Specifies the number of workers to be used when loading the dataset.
- **weight_decay**: *float, default=5e-4*\
  Specifies the weight decay to be used during training.
- **momentum**: *float, default=0.9*\
  Specifies the momentum to be used for optimizer during training.
- **scale**: *float, default=1.0*\
  Specifies the downsampling scale factor between input image and output heatmap.
- **wh_weight**: *float, default=0.1*\
  Specifies the weight of the width/height loss during training.
- **center_reg_weight**: *float, default=1.0*\
  Specifies the weight of the center loss during training.
- **warmup_epochs**: *int, default=0*\
  Specifies the number of epochs at the beginning of training, during which the learning rate is annealed until it reaches the specified *lr*.
- **lr_decay_epoch**: *str, default='80,100'*\
  Specifies the epochs at which the learning rate is dropped during training.
  Must be a string using commas as a delimiter. Ignored if *lr_decay_period* is set to a value larger than 0.
- **lr_decay**: *float, default=0.1*\
  Specifies the rate by which the learning rate drops during training.
- **flip_validation**: *bool, default=False*\
  Specifies whether to flip images during evaluation to increase performance.
  Increases evaluation time.


#### `YOLOv3DetectorLearner.fit`
```python
YOLOv3DetectorLearner.fit(self, dataset, val_dataset, verbose)
```

This method is used to train the algorithm on a `DetectionDataset` or `ExternalDataset` dataset and also performs evaluation
on a validation set using the trained model. Returns a dictionary containing stats regarding the training process.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
- **val_dataset**: *object, default=None*\
  Object that holds the validation dataset.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.

#### `YOLOv3DetectorLearner.eval`
```python
YOLOv3DetectorLearner.eval(self, dataset, use_subset, subset_size, verbose)
```

Performs evaluation on a dataset or a subset of a dataset.

Parameters:

- **dataset**: *object*\
  Object that holds dataset to perform evaluation on.
- **use_subset**: *bool, default=False*\
  If True, evaluation is perform on a random subset of **subset_size** images from the **dataset**.
- **subset_size**: *int, default=250*\
  Only used if **used_subset=True**. Specifies the number of images to perform evaluation on.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `YOLOv3DetectorLearner.infer`
```python
YOLOv3DetectorLearner.infer(self, img, threshold, keep_size)
```

Performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.2*\
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
- **keep_size**: *bool, default=False*\
  Specifies whether to resize the input image to *self.img_size* or keep original image dimensions.

#### `YOLOv3DetectorLearner.save`
```python
YOLOv3DetectorLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `YOLOv3DetectorLearner.load`
```python
YOLOv3DetectorLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `YOLOv3DetectorLearner.download`
```python
YOLOv3DetectorLearner.download(self, path, mode, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
- **mode**: *{'pretrained', 'images', 'test_data'}, default='pretrained'*\
  If *'pretrained'*, downloads a pretrained detector model. If *'images'*, downloads an image to perform inference on. If
  *'test_data'* downloads a dummy dataset for testing purposes.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.
  
#### Examples

* **Training example using an `ExternalDataset`**.
  To train properly, the backbone weights are downloaded automatically in the `temp_path`. Default backbone is 'darknet53'.
  The VOC and COCO datasets are supported as ExternalDataset types. This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.

  ```python
  from opendr.perception.object_detection_2d import YOLOv3DetectorLearner
  from opendr.engine.datasets import ExternalDataset
  
  dataset = ExternalDataset(data_root, 'voc')
  val_dataset = ExternalDataset(data_root, 'voc')

  yolo = YOLOv3DetectorLearner(device=device, batch_size=batch_size, lr=lr,
                               val_after=val_after,
                               epochs=n_epochs, backbone=backbone)

  yolo.fit(dataset, val_dataset)
  yolo.save('./trained_models/yolo_saved_model')
  ```
  
  Training with `DetetionDataset` types is also supported. Example using the `WiderPersonDataset` (assuming data has been downloaded in `data_root` folder):
  ```python
  from opendr.perception.object_detection_2d import YOLOv3DetectorLearner
  from opendr.perception.object_detection_2d import WiderPersonDataset
  
  dataset = WiderPersonDataset(root=data_root, splits=['train'])
  val_dataset = WiderPersonDataset(root=data_root, splits=['val'])

  yolo = YOLOv3DetectorLearner(device=device, batch_size=batch_size, lr=lr,
                               val_after=val_after,
                               epochs=n_epochs, backbone=backbone)

  yolo.fit(dataset, val_dataset)
  yolo.save('./trained_models/yolo_saved_model')
  ```
  
  Custom datasets are supported by inheriting the `DetectionDataset` class.

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import YOLOv3DetectorLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  yolo = YOLOv3DetectorLearner(device=device)
  yolo.download('.', mode="pretrained")
  yolo.load('./yolo_default', verbose=True)

  yolo.download('.', mode='images', verbose=True)
  img = Image.open('./cat.jpg')

  boxes = yolo.infer(img)
  draw_bounding_boxes(img.opencv(), boxes, class_names=yolo.classes, show=True)
  ```
  

#### Performance Evaluation

In terms of speed, the performance of YOLOv3 is summarized in the table below (in FPS).

| Method | RTX 2070 | TX2 | AGX |
|---------|----------|-----|-----|
| YOLOv3 | 50 | 9 | 16 |

Apart from the inference speed, we also report the memory usage, as well as energy consumption on a reference platform in the Table below.
The measurement was made on a Jetson TX2 module.

| Method  | Memory (MB) | Energy (Joules)  - Total per inference  |
|-------------------|---------|-------|
| YOLOv3 | 5219 | 11.88  | 


Finally, we measure the performance on the COCO dataset, using the corresponding metrics.

| Metric   | YOLOv3   |
|---------|-----------|
| mAP     | 36.0      |
| AP@0.5  | 57.2      |
| AP@0.75 | 38.7      |
| mAP (S) | 17.3      |
| mAP (M) | 38.8      |
| mAP (L) | 52.3      |
| AR      | 44.5      |
| AR (S)  | 23.6      |
| AR (M)  | 47.2      |
| AR (L)  | 62.5      |

The platform compatibility evaluation is also reported below:

| Platform  | Compatibility Evaluation |
| ----------------------------------------------|-------|
| x86 - Ubuntu 20.04 (bare installation - CPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (bare installation - GPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (pip installation)         | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (CPU docker)               | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (GPU docker)               | :heavy_check_mark:   |
| NVIDIA Jetson TX2                             | :heavy_check_mark:   |
| NVIDIA Jetson Xavier AGX                      | :heavy_check_mark:   |
| NVIDIA Jetson Xavier NX                       | :heavy_check_mark:   |

#### References
<a name="yolo-1" href="https://arxiv.org/abs/1804.02767">[1]</a> YOLOv3: An Incremental Improvement,
[arXiv](https://arxiv.org/abs/1804.02767).
