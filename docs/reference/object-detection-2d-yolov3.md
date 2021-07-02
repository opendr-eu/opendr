## YOLOv3DetectorLearner module

The *yolov3* module contains the *YOLOv3DetectorLearner* class, which inherits from the abstract class *Learner*.

### Class YOLOv3DetectorLearner
Bases: `engine.learners.Learner`

The *YOLOv3DetectorLearner* class is a wrapper of the SSD detector[[1]](#yolo-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/yolo/yolo3.py).
It can be used to perform object detection on images (inference) as well as train new object detection models.

The [YOLOv3DetectorLearner](#src.opendr.perception.object_detection_2d.yolov3.yolov3_learner.py) class has the following 
public methods:

#### `YOLOv3DetectorLearner` constructor
```python
YOLOv3DetectorLearner(self, lr, epochs, batch_size, device, backbone, img_size,
                      lr_schedule, temp_path, checkpoint_after_iter, checkpoint_load_iter,
                      val_after, log_after, num_workers, weight_decay, momentum, mixup,
                      no_mixup_epochs, label_smoothing, random_shape, warmup_epochs, lr_decay_period,
                      lr_decay_epoch, lr_decay)
```

Constructor parameter explanation:
- **lr**: *float, default=0.0001*
  Specifies the initial learning rate to be used during training.
  
- **epochs**: *int, default=100*
  Specifies the number of epochs to be used during training.
  
- **batch_size**: *int, default=2*
  Specifies the batch size to be used during training.
  
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device to be used.
  
- **backbone**: *{'darknet53', 'mobilenet1.0', 'mobilenet0.25'}, default='darknet53'*
  Specifies the backbone architecture. 
  
- **img_size**: *int, default=416* 
  Specifies the image size to be used during training. Must be a multiple of *32*.
  
- **lr_schedule**: *{'step', 'cosine', 'poly'}, default='step'* 
  Specifies the desired learning rate schedule.
  
- **temp_path**: *str, default=''*
  Specifies a path to be used for storage of checkpoints during training. 
  
- **checkpoint_after_iter**: *int, default=0*
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
  
- **checkpoint_load_iter**: *int, default=0*
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
  
- **val_after**: *int, default=5*
  Epoch interval between evaluations during training.
  
- **log_after**: *int, default=100*
  Specifies interval (in iterations/batches) between information logging on *stdout*.
  
- **num_workers**: *int, default=8* Specifies the number of workers to be used when loading the dataset.
  
- **weight_decay**: *float, default=5e-4*
  Specifies the weight decay to be used during training. 
  
- **momentum**: *float, default=0.9*
  Specifies the momentum to be used for optimizer during training.
  
- **scale**: *float, default=1.0*
  Spevifies the downsampling scale factor between input image and output heatmap.
  
- **wh_weight**: *float, default=0.1* 
  Specifies the weight of the width/height loss during training.

- **center_reg_weight**: *float, default=1.0* 
  Specifies the weight of the center loss during training.

- **warmup_epochs**: *int, default=0* Specifies the number of epochs at the beginning of training, during which the learning 
  rate is annealed until it reaches the specified *lr*.
  
- **lr_decay_epoch**: *str, default='80,100'* Specifies the epochs at which the learning rate is dropped during training. 
  Must be a string using commas as a delimiter. Ignored if *lr_decay_period* is set to a value larger than 0.
  
- **lr_decay**: *float, default=0.1* Specifies the rate by which the learning rate drops during training.

- **flip_validation**: *bool, default=False* Specifies whether to flip images during evaluation to increase performance. 
  Increases evaluation time.
  
  
#### `YOLOv3DetectorLearner.fit`
```python
YOLOv3DetectorLearner.fit(self, dataset, val_dataset, verbose)
```

This method is used to train the algorithm on a `DetectionDataset` or `ExternalDataset` dataset and also performs evaluation 
on a validation set using the trained model. Returns a dictionary containing stats regarding the training process.
Parameters:
- **dataset**: *object*
  Object that holds the training dataset.
  
- **val_dataset**: *object, default=None*
  Object that holds the validation dataset.
  
- **verbose**: *bool, default=True*
  If True, enables maximum verbosity.
  
#### `YOLOv3DetectorLearner.eval`
```python
YOLOv3DetectorLearner.eval(self, dataset, use_subset, subset_size, verbose)
```

Performs evaluation on a dataset or a subset of a dataset.
Parameters:
- **dataset**: *object*
  Object that holds dataset to perform evaluation on.
  
- **use_subset**: *bool, default=False*
  If True, evaluation is perform on a random subset of **subset_size** images from the **dataset**.
  
- **subset_size**: *int, default=250*
  Only used if **used_subset=True**. Specifies the number of images to perform evaluation on.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `YOLOv3DetectorLearner.infer`
```python
YOLOv3DetectorLearner.infer(self, img, threshold=0.2, keep_size=False)
```

Performs inference on a single image.
Parameters:
- **img**: *object*
  Object of type engine.data.Image.
    
- **threshold**: *float, default=0.2*
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
  
- **keep_size**: *bool, default=False* 
  Specifies whether to resize the input image to *self.img_size* or keep original image dimensions.
  
#### `YOLOv3DetectorLearner.save`
```python
YOLOv3DetectorLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.
Parameters:
- **path**: *str*
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `YOLOv3DetectorLearner.load`
```python
YOLOv3DetectorLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:
- **path**: *str*
  Specifies the folder where the model will be loaded from.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `YOLOv3DetectorLearner.download`
```python
YOLOv3DetectorLearner.download(self, path, mode, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.
Parameters:
- **path**: *str, default=None*
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
  
- **mode**: *{'pretrained', 'images', 'test_data'}, default='pretrained'*
  If *'pretrained'*, downloads a pretrained detector model. If *'images'*, downloads an image to perform inference on. If 
  *'test_data'* downloads a dummy dataset for testing purposes.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
- **url**: *str, default=OpenDR FTP URL* 
  URL of the FTP server.
  
#### References
<a name="yolo-1" href="https://arxiv.org/abs/1804.02767">[1]</a> YOLOv3: An Incremental Improvement,
[arXiv](https://arxiv.org/abs/1804.02767).