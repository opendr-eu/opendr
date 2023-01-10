## RetinaFace module

The *retinaface* module contains the *RetinaFaceLearner* class, which inherits from the abstract class *Learner*.

### Class RetinaFaceLearner
Bases: `engine.learners.Learner`

The *RetinaFaceLearner* class is a wrapper of the RetinaFace detector[[1]](#retinaface-1) implementation found on
[deepinsight implementation](https://www.github.com/deepinsight/insightface).
It can be used to perform face detection on images (inference) as well as train new face detection models.

The [RetinaFaceLearner](/src/opendr/perception/object_detection_2d/retinaface/retinaface_learner.py) class has the following
public methods:

#### `RetinaFaceLearner` constructor
```python
RetinaFaceLearner(backbone, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter,
                 lr_steps, epochs, momentum, weight_decay, log_after, prefix,
                 shuffle, flip, val_after, temp_path, device)
```

Constructor parameters:

- **backbone**: *{'resnet', 'mnet}', default='resnet'*\
  Specifies the backbone architecture. Only *'resnet'* is supported for training, but a
  *'mnet'* model is supported for inference.
- **lr**: *float, default=0.0001*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=2*\
  Specifies the batch size to be used during training.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
- **lr_steps**: *str, default='0'*\
  Specifies the epochs at which the learning rate will be adjusted. Must be an *str* using commas as the delimiter.
- **epochs**: *int, default=100*\
  Specifies the number of epochs to be used during training.
- **momentum**: *float, default=0.9*\
  Specifies the momentum to be used for SGD during training.
- **weight_decay**: *float, default=5e-4*\
  Specifies the weight decay to be used during training.
- **log_after**: *int, default=20*\
  Specifies interval (in iterations/batches) between information logging on *stdout*.
- **prefix**: *str, default=''*\
  Sets an experiment name to be used during checkpoint saving.
- **shuffle**: *bool, default=True*\
  Specifies whether to shuffle data during training.
- **flip**: *bool, default=False*\
  Specifies whether to perform image flipping during training.
- **val_after**: *int, default=5*\
  Epoch interval between evaluations during training.
- **temp_path**: *str, default=''*\
  Specifies a path to be used for storage of checkpoints during training. A roidb file is also stored in the specified
  directory during training.
- **device**: *{'cuda', 'cpu'}, default='cuda'*\
  Specifies the device to be used.

#### `RetinaFaceLearner.fit`
```python
RetinaFaceLearner.fit(self, dataset, val_dataset, from_scratch, silent, verbose)
```

This method is used to train the algorithm on the `WiderFaceDetection` dataset and also performs evaluation on a validation
set using the trained model. Returns a dictionary containing stats regarding the training process.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset. Must be of `WiderFaceDetection` type.
- **val_dataset**: *object, default=None*\
  Object that holds the validation dataset.
- **from_scratch**: *bool, default=False*\
  Specifies whether to train from scratch or to download and use a pretrained backbone.
- **silent**: *bool, default=False*\
  If set to True, disables all printing to STDOUT, defaults to False.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.

#### `RetinaFaceLearner.eval`
```python
RetinaFaceLearner.eval(self, dataset, verbose, use_subset, subset_size, pyramid, flip)
```

Performs evaluation on a dataset or a subset of a dataset.

Parameters:

- **dataset**: *object*\
  Object that holds dataset to perform evaluation on.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.
- **use_subset**: *bool, default=False*\
  If True, evaluation is perform on a random subset of **subset_size** images from the **dataset**.
- **subset_size**: *int, default=250*\
  Only used if **used_subset=True**. Specifies the number of images to perform evaluation on.
- **flip**: *bool, default=True*\
  Specifies whether to perform image flipping during evaluation to increase performance. Increases evaluation time.
- **pyramid**: *bool, default=True*\
  Specifies whether to use an image pyramid during evaluation to increase performance. Increases evaluation time.

#### `RetinaFaceLearner.infer`
```python
RetinaFaceLearner.infer(self, img, threshold, nms_threshold, scales, mask_thresh)
```

Performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.8*\
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
- **nms_threshold**: *float, default=0.4*\
  Defines the NMS threshold.
- **scales**: *list, default=[1024, 1980]*\
  Defines the image scales to perform detection on.
- **mask_thresh**: *float, default=0.8*\
  If using an *'mnet'* model which can detect masked faces, defines the mask detection threshold. Bounding boxes with mask
  confidence under this value are deemed to be maskless faces.

#### `RetinaFaceLearner.save`
```python
RetinaFaceLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `RetinaFaceLearner.load`
```python
RetinaFaceLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `RetinaFaceLearner.download`
```python
RetinaFaceLearner.download(self, path, mode, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., data for training, pretrained models and backbones as
well as test data.

Parameters:
- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
- **mode**: *{'pretrained', 'images', 'backbone', 'annotations'}, default='pretrained'*\
  If *'pretrained'*, downloads a pretrained detector model based on *self.backbone*. If *'backbone'*, downloads a pretrained
  backbone network based on *self.backbone*, used to train a model from pretrained weights. If *'images'*, downloads an
  image to perform inference on. If *'annotations'*, downloads additional landmark annotations to be used during training.
- **verbose**: *bool, default=True*\
  If True, maximum verbosity if enabled.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.

#### Examples

* **Training example**.
  To train properly, the backbone weights are downloaded automatically in the `temp_path`.
  The WIDER Face detection dataset is supported for training, implemented as a `DetectionDataset` subclass. This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.

  ```python
  from opendr.perception.object_detection_2d import RetinaFaceLearner, WiderFaceDataset
  from opendr.engine.datasets import ExternalDataset

  dataset = WiderFaceDataset(root=data_root, splits=['train'])

  face_learner = RetinaFaceLearner(backbone='resnet', prefix='retinaface_resnet50',
                                     epochs=n_epochs, log_after=10, flip=False, shuffle=False,
                                     lr=lr, lr_steps='55,68,80', weight_decay=5e-4,
                                     batch_size=4, val_after=val_after,
                                     temp_path='temp_retinaface', checkpoint_after_iter=1)

  face_learner.fit(dataset, val_dataset=dataset, verbose=True)
  face_learner.save('./trained_models/retinaface_resnet50')
  ```

  Custom datasets are supported by inheriting the `DetectionDataset` class.

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import RetinaFaceLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  learner = RetinaFaceLearner(backbone=backbone, device=device)
  learner.download('.', mode='pretrained')
  learner.load('./retinaface_{}'.format(backbone))

  learner.download('.', mode='images')
  img = Image.open('./cov4.jpg')
  bounding_boxes = learner.infer(img)

  img = draw_bounding_boxes(img.opencv(), bounding_boxes, learner.classes, show=True)
  ```

#### Performance Evaluation

In terms of speed, the performance of RetinaFace is summarized in the table below (in FPS).

| Variant | RTX 2070 | TX2 | AGX |
|---------|----------|-----|-----|
| RetinaFace | 47 | 3 | 8 |
| RetinaFace-MobileNet | 114 | 13 | 18 |

Apart from the inference speed, we also report the memory usage, as well as energy consumption on a reference platform in the Table below.
The measurement was made on a Jetson TX2 module.

| Variant  | Memory (MB) | Energy (Joules)  - Total per inference  |
|-------------------|---------|-------|
| RetinaFace | 4443 | 21.83  |
| RetinaFace-MobileNet     | 4262 | 8.73  |

Finally, we measure the recall on the WIDER face validation subset at 87.83%.
Note that RetinaFace can make use of image pyramids and horizontal flipping to achieve even better recall at the cost of additional computations.
For the MobileNet version, recall drops to 77.81%.

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
<a name="retinaface-1" href="https://arxiv.org/abs/1905.00641">[1]</a> RetinaFace: Single-stage Dense Face Localisation in the Wild,
[arXiv](https://arxiv.org/abs/1905.00641).

