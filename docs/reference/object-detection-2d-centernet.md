## CenterNetDetectorLearner module

The *centernet* module contains the *CenterNetDetectorLearner* class, which inherits from the abstract class *Learner*.

### Class CenterNetDetectorLearner
Bases: `engine.learners.Learner`

The *CenterNetDetectorLearner* class is a wrapper of the CenterNet detector[[1]](#centernet-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/center_net/center_net.py).
It can be used to perform object detection on images (inference) as well as train new object detection models.

The [CenterNetDetectorLearner](/src/opendr/perception/object_detection_2d/centernet/centernet_learner.py) class has the following
public methods:

#### `CenterNetDetectorLearner` constructor
```python
CenterNetDetectorLearner(self, lr, epochs, batch_size, device, backbone, img_size,
                         lr_schedule, temp_path, checkpoint_after_iter, checkpoint_load_iter,
                         val_after, log_after, num_workers, weight_decay, momentum,
                         scale, topk, wh_weight, center_reg_weight,
                         lr_decay_epoch, lr_decay, warmup_epochs, flip_validation)
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
- **backbone**: *{'resnet50_v1b'}, default='resnet50_v1b'*\
  Specifies the backbone architecture. Currently only *'resnet50_v1b'* is supported.
- **img_size**: *int, default=512*\
  Specifies the image size to be used during training.
- **lr_schedule**: *{'constant', 'step', 'linear', 'poly', 'cosine'}, default='step'*\
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
  Specifies interval (in iterations/batches) between information logging on *stdout*.\
- **num_workers**: *int, default=8* Specifies the number of workers to be used when loading the dataset.
- **weight_decay**: *float, default=5e-4*\
  Specifies the weight decay to be used during training.
- **momentum**: *float, default=0.9*\
  Specifies the momentum to be used for optimizer during training.
- **mixup**: *bool default=False*\
  Specifies whether to use an image mixing strategy during training.
- **no_mixup_epochs**: *int, default=0*\
  If *mixup* is True, specifies the number of epochs for which mixup is disabled in
  the beginning of the training process.
- **label_smoothing**: *bool default=False*\
  Specifies whether to use label smoothing during training.
- **random_shape**: *bool, default=False*\
  Specifies whether to resize input images during training to random multiples of *32*.
- **warmup_epochs**: *int, default=0*\
  Specifies the number of epochs at the beginning of training, during which the learning rate is annealed until it reaches the specified *lr*.
- **lr_decay_period**: *int, default=0*\
  Specifies the interval at which the learning rate drops.
- **lr_decay_epoch**: *str, default='80,100'*\
  Specifies the epochs at which the learning rate is dropped during training.
  Must be a string using commas as a delimiter. Ignored if *lr_decay_period* is set to a value larger than 0.
- **lr_decay**: *float, default=0.1*\
  Specifies the rate by which the learning rate drops during training.



#### `CenterNetDetectorLearner.fit`
```python
CenterNetDetectorLearner.fit(self, dataset, val_dataset, verbose)
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

#### `CenterNetDetectorLearner.eval`
```python
CenterNetDetectorLearner.eval(self, dataset, use_subset, subset_size, verbose)
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

#### `CenterNetDetectorLearner.infer`
```python
CenterNetDetectorLearner.infer(self, img, threshold, keep_size)
```

Performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.2*\
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
- **keep_size**: *bool, default=False*\
  Specifies whether to resize the input image to *self.img_size* or keep original image dimensions.

#### `CenterNetDetectorLearner.save`
```python
CenterNetDetectorLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `CenterNetDetectorLearner.load`
```python
CenterNetDetectorLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `CenterNetDetectorLearner.download`
```python
CenterNetDetectorLearner.download(self, path, mode, verbose, url)
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
  To train properly, the backbone weights are downloaded automatically in the `temp_path`. Default backbone is
  'resnet50_v1b'.
  The VOC and COCO datasets are supported as ExternalDataset types. This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.

  ```python
  from opendr.perception.object_detection_2d import CenterNetDetectorLearner
  from opendr.engine.datasets import ExternalDataset

  centernet = CenterNetDetectorLearner(device=device, batch_size=batch_size,
                                       lr=lr, val_after=val_after,
                                       epochs=n_epochs, backbone=backbone)

  dataset = ExternalDataset(data_root, 'voc')
  val_dataset = ExternalDataset(data_root, 'voc')
  centernet.fit(dataset, val_dataset)
  centernet.save('./trained_models/centernet_model')
  ```
  
  Training with `DetetionDataset` types is also supported. Example using the `WiderPersonDataset` (assuming data has been downloaded in `data_root` folder):
  ```python
  from opendr.perception.object_detection_2d import CenterNetDetectorLearner
  from opendr.perception.object_detection_2d import WiderPersonDataset

  centernet = CenterNetDetectorLearner(device=device, batch_size=batch_size,
                                       lr=lr, val_after=val_after,
                                       epochs=n_epochs, backbone=backbone)

  dataset = WiderPersonDataset(root=data_root, splits=['train'])
  val_dataset = WiderPersonDataset(root=data_root, splits=['val'])
  centernet.fit(dataset, val_dataset)
  centernet.save('./trained_models/centernet_model')
  ```
  
  Custom datasets are supported by inheriting the `DetectionDataset` class.

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import CenterNetDetectorLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  centernet = CenterNetDetectorLearner(device=device)
  centernet.download(".", mode="pretrained")
  centernet.load("./centernet_default", verbose=True)

  centernet.download(".", mode="images")
  img = Image.open("./bicycles.jpg")

  boxes = centernet.infer(img)
  draw_bounding_boxes(img.opencv(), boxes, class_names=centernet.classes, show=True)
  ```
  
#### Performance Evaluation

In terms of speed, the performance of CenterNet is summarized in the table below (in FPS).

| Method | RTX 2070 | TX2 | AGX |
|---------|----------|-----|-----|
| CenterNet | 88 | 19 | 14 |

Apart from the inference speed, we also report the memory usage, as well as energy consumption on a reference platform in the Table below.
The measurement was made on a Jetson TX2 module.

| Method  | Memory (MB) | Energy (Joules)  - Total per inference  |
|-------------------|---------|-------|
| CenterNet | 4784 | 12.01  | 


Finally, we measure the performance on the COCO dataset, using the corresponding metrics.

| Metric   | CenterNet |
|---------|-----------|
| mAP     | 7.5       |
| AP@0.5  | 24.5      |
| AP@0.75 | 1.0       |
| mAP (S) | 0.9       |
| mAP (M) | 5.4       |
| mAP (L) | 17.0      |
| AR      | 14.9      |
| AR (S)  | 2.9       |
| AR (M)  | 12.9      |
| AR (L)  | 30.3      |

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
<a name="centernet-1" href="https://arxiv.org/abs/1904.08189">[1]</a> CenterNet: Keypoint Triplets for Object Detection,
[arXiv](https://arxiv.org/abs/1904.08189).
