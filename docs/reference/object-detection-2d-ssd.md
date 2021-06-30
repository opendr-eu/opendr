## SingleShotDetector module

The *ssd* module contains the *SingleShotDetectorLearner* class, which inherits from the abstract class *Learner*.

### Class SingleShotDetectorLearner
Bases: `engine.learners.Learner`

The *SingleShotDetectorLearner* class is a wrapper of the SSD detector[[1]](#ssd-1)
[GluonCV implementation](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/ssd/ssd.py).
It can be used to perform face detection on images (inference) as well as train new face detection models.

The [SingleShotDetectorLearner](#src.opendr.perception.object_detection_2d.ssd.ssd_learner.py) class has the following 
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

Constructor parameter explanation:
- **lr**: *float, default=0.0001*
  Specifies the initial learning rate to be used during training.
  
- **epochs**: *int, default=100*
  Specifies the number of epochs to be used during training.
  
- **batch_size**: *int, default=2*
  Specifies the batch size to be used during training.
  
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device to be used.
  
- **backbone**: *{'vgg16_atrous', 'resnet50_v1', 'mobilenet1.0', 'mobilenet0.25', 'resnet34_v1b'}, default='vgg16_atrous'*
  Specifies the backbone architecture. 
  
- **img_size**: *{512, 300}, default=512* 
  Specifies the image size to be used during training.
  
- **lr_schedule**: *{'', 'warmup'}, default=''* Specifies the desired learning rate schedule. If *'warmup'*, training starts 
  with a lower learning rate than the specified *lr* and gradually reaches it within the first epochs of training.
  
- **temp_path**: *str, default=''*
  Specifies a path to be used for storage of checkpoints during training. A roidb file is also stored in the specified 
  directory during training.
  
- **checkpoint_after_iter**: *int, default=0*
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
  
- **checkpoint_load_iter**: *int, default=0*
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
  
- **val_after**: *int, default=5*
  Epoch interval between evaluations during training.
  
- **log_after**: *int, default=20*
  Specifies interval (in iterations/batches) between information logging on *stdout*.
  
- **num_workers**: *int, default=8* Specifies the number of workers to be used when loading the dataset.
  
- **weight_decay**: *float, default=5e-4*
  Specifies the weight decay to be used during training. 
  
- **momentum**: *float, default=0.9*
  Specifies the momentum to be used for optimizer during training.
  
  
#### `SingleShotDetectorLearner.fit`
```python
SingleShotDetectorLearner.fit(self, dataset, val_dataset, verbose)
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
  
#### `SingleShotDetectorLearner.eval`
```python
SingleShotDetectorLearner.eval(self, dataset, use_subset, subset_size, verbose)
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
  
#### `SingleShotDetectorLearner.infer`
```python
SingleShotDetectorLearner.infer(self, img, threshold=0.2, keep_size=False)
```

Performs inference on a single image.
Parameters:
- **img**: *object*
  Object of type engine.data.Image.
    
- **threshold**: *float, default=0.2*
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
  
- **keep_size**: *bool, default=False* 
  Specifies whether to resize the input image to *self.img_size* or keep original image dimensions.
  
#### `SingleShotDetectorLearner.save`
```python
SingleShotDetectorLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.
Parameters:
- **path**: *str*
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `SingleShotDetectorLearner.load`
```python
SingleShotDetectorLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:
- **path**: *str*
  Specifies the folder where the model will be loaded from.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `SingleShotDetectorLearner.download`
```python
SingleShotDetectorLearner.download(self, path, mode, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.
Parameters:
- **path**: *str, default=None*
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
  
- **mode**: *{'pretrained', 'images', 'test_data'}, default='pretrained'*
  If *'pretrained'*, downloads a pretrained detector model. If *'images'*, downloads an image to perform inference on. If 
  *'test_data'* downloads a dummy dataset for testing purposes. 
  
#### References
<a name="ssd-1" href="https://arxiv.org/abs/1512.02325">[1]</a> SSD: Single Shot MultiBox Detector,
[arXiv](https://arxiv.org/abs/1512.02325).