## SingleShotDetector module

The *ssd* module contains the *SingleShotDetectorLearner* class, which inherits from the abstract class *Learner*.

### Class SingleShotDetectorLearner
Bases: `engine.learners.Learner`

The *SingleShotDetectorLearner* class is a wrapper of the SSD detector[[1]](#ssd-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/ssd/ssd.py).
It can be used to perform object detection on images (inference) as well as train new object detection models.

The [SingleShotDetectorLearner](/src/opendr/perception/object_detection_2d/ssd/ssd_learner.py) class has the following
public methods:

#### `SingleShotDetectorLearner` constructor
```python
SingleShotDetectorLearner(self, lr, epochs, batch_size,
                          device, backbone,
                          img_size, lr_schedule, temp_path,
                          checkpoint_after_iter, checkpoint_load_iter,
                          val_after, log_after, num_workers,
                          weight_decay, momentum)
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
- **backbone**: *{'vgg16_atrous', 'resnet50_v1', 'mobilenet1.0', 'mobilenet0.25', 'resnet34_v1b'}, default='vgg16_atrous'*\
  Specifies the backbone architecture.
- **img_size**: *{512, 300}, default=512*\
  Specifies the image size to be used during training.
- **lr_schedule**: *{'', 'warmup'}, default=''*\
  Specifies the desired learning rate schedule. If *'warmup'*, training starts with a lower learning rate than the specified *lr* and gradually reaches it within the first epochs of training.
- **temp_path**: *str, default=''*\
  Specifies a path to be used for storage of checkpoints during training.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
- **val_after**: *int, default=5*\
  Epoch interval between evaluations during training.
- **log_after**: *int, default=20*\
  Specifies interval (in iterations/batches) between information logging on *stdout*.
- **num_workers**: *int, default=8*\
  Specifies the number of workers to be used when loading the dataset.
- **weight_decay**: *float, default=5e-4*\
  Specifies the weight decay to be used during training.
- **momentum**: *float, default=0.9*\
  Specifies the momentum to be used for optimizer during training.


#### `SingleShotDetectorLearner.fit`
```python
SingleShotDetectorLearner.fit(self, dataset, val_dataset, verbose)
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

#### `SingleShotDetectorLearner.eval`
```python
SingleShotDetectorLearner.eval(self, dataset, use_subset, subset_size, verbose)
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

#### `SingleShotDetectorLearner.infer`
```python
SingleShotDetectorLearner.infer(self, img, threshold, keep_size)
```

Performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.2*\
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
- **keep_size**: *bool, default=False*\
  Specifies whether to resize the input image to *self.img_size* or keep original image dimensions.

#### `SingleShotDetectorLearner.save`
```python
SingleShotDetectorLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `SingleShotDetectorLearner.load`
```python
SingleShotDetectorLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.

#### `SingleShotDetectorLearner.download`
```python
SingleShotDetectorLearner.download(self, path, mode, verbose, url)
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
  To train properly, the backbone weights are downloaded automatically in the `temp_path`. Default backbone is 'vgg16_atrous'.
  The VOC and COCO datasets are supported as ExternalDataset types. This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.

  ```python
  from opendr.perception.object_detection_2d import SingleShotDetectorLearner
  from opendr.engine.datasets import ExternalDataset
  
  dataset = ExternalDataset(data_root, 'voc')
  val_dataset = ExternalDataset(data_root, 'voc')

  ssd = SingleShotDetectorLearner(device=device, batch_size=batch_size, lr=lr,                                                val_after=val_after,
                                  checkpoint_load_iter=resume_from, epochs=n_epochs,
                                  checkpoint_after_iter=checkpoint_freq)

  ssd.fit(dataset, val_dataset)
  ssd.save('./trained_models/ssd_saved_model')
  ```
  
  Training with `DetetionDataset` types is also supported. Example using the `WiderPersonDataset` (assuming data has been downloaded in `data_root` folder):
  ```python
  from opendr.perception.object_detection_2d import SingleShotDetectorLearner
  from opendr.perception.object_detection_2d.datasets import WiderPersonDataset
  
  dataset = WiderPersonDataset(root=data_root, splits=['train'])
  val_dataset = WiderPersonDataset(root=data_root, splits=['val'])

  ssd = SingleShotDetectorLearner(device=device, batch_size=batch_size, lr=lr,                                                val_after=val_after,
                                  checkpoint_load_iter=resume_from, epochs=n_epochs,
                                  checkpoint_after_iter=checkpoint_freq)

  ssd.fit(dataset, val_dataset)
  ssd.save('./trained_models/ssd_saved_model')
  ```
  
  Custom datasets are supported by inheriting the `DetectionDataset` class.

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import SingleShotDetectorLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  ssd = SingleShotDetectorLearner(device=device)
  ssd.download('.', mode='pretrained')
  ssd.load('./ssd_default_person', verbose=True)

  ssd.download('.', mode='images')
  img = Image.open('./people.jpg')

  boxes = ssd.infer(img)
  draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)
  ```
  

#### Performance Evaluation

In terms of speed, the performance of SSD is summarized in the table below (in FPS).

| Method | RTX 2070 | TX2 | AGX |
|---------|----------|-----|-----|
| SSD | 85 | 16 | 27 |

Apart from the inference speed, we also report the memory usage, as well as energy consumption on a reference platform in the Table below.
The measurement was made on a Jetson TX2 module.

| Method  | Memory (MB) | Energy (Joules)  - Total per inference  |
|-------------------|---------|-------|
| SSD | 4583 | 2.47  | 


Finally, we measure the performance on the COCO dataset, using the corresponding metrics.

| Metric   | SSD      |
|---------|-----------|
| mAP     | 27.4      |
| AP@0.5  | 45.6      |
| AP@0.75 | 28.9      |
| mAP (S) | 2.8       |
| mAP (M) | 25.8      |
| mAP (L) | 42.9      |
| AR      | 36.3      |
| AR (S)  | 4.5       |
| AR (M)  | 37.5      |
| AR (L)  | 53.7      |

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
<a name="ssd-1" href="https://arxiv.org/abs/1512.02325">[1]</a> SSD: Single Shot MultiBox Detector,
[arXiv](https://arxiv.org/abs/1512.02325).
