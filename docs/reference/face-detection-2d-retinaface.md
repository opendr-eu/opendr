## RetinaFace module

The *retinaface* module contains the *RetinaFaceLearner* class, which inherits from the abstract class *Learner*.

### Class RetinaFaceLearner
Bases: `engine.learners.Learner`

The *RetinaFaceLearner* class is a wrapper of the RetinaFace detector[[1]](#retinaface-1) implementation found on 
[deepinsight implementation](https://www.github.com/deepinsight/insightface).
It can be used to perform face detection on images (inference) as well as train new face detection models.

The [RetinaFaceLearner](#src.opendr.perception.object_detection_2d.retinaface.retinaface_learner.py) class has the following 
public methods:

#### `RetinaFaceLearner` constructor
```python
RetinaFaceLearner(backbone, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter,
                 lr_steps, epochs, momentum, weight_decay, log_after, prefix,
                 shuffle, flip, val_after, temp_path, device)
```

Constructor parameter explanation:
- **backbone**: *{'resnet', 'mnet}', default='resnet'*
  Specifies the backbone architecture. Only *'resnet'* is supported for training, but a 
  *'mnet'* model is supported for inference.
  
- **lr**: *float, default=0.0001*
  Specifies the initial learning rate to be used during training.
  
- **batch_size**: *int, default=2*
  Specifies the batch size to be used during training.
  
- **checkpoint_after_iter**: *int, default=0*
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
  
- **checkpoint_load_iter**: *int, default=0*
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
  
- **lr_steps**: *str, default='0'*
  Specifies the epochs at which the learning rate will be adjusted. Must be an *str* using commas as the delimiter.
  
- **epochs**: *int, default=100*
  Specifies the number of epochs to be used during training.
  
- **momentum**: *float, default=0.9*
  Specifies the momentum to be used for SGD during training.
  
- **weight_decay**: *float, default=5e-4*
  Specifies the weight decay to be used during training.
  
- **log_after**: *int, default=20*
  Specifies interval (in iterations/batches) between information logging on *stdout*.
  
- **prefix**: *str, default=''*
  Sets an experiment name to be used during checkpoint saving.
  
- **shuffle**: *bool, default=True*
  Specifies whether to shuffle data during training.
  
- **flip**: *bool, default=False*
  Specifies whether to perform image flipping during training.
  
- **val_after**: *int, default=5*
  Epoch interval between evaluations during training.
  
- **temp_path**: *str, default=''*
  Specifies a path to be used for storage of checkpoints during training. A roidb file is also stored in the specified 
  directory during training.
  
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device to be used.
  
#### `RetinaFaceLearner.fit`
```python
RetinaFaceLearner.fit(self, dataset, val_dataset, from_scratch, silent, verbose)
```

This method is used to train the algorithm on the `WiderFaceDetection` dataset and also performs evaluation on a validation 
set using the trained model. Returns a dictionary containing stats regarding the training process.
Parameters:
- **dataset**: *object*
  Object that holds the training dataset. Must be of `WiderFaceDetection` type.
  
- **val_dataset**: *object, default=None*
  Object that holds the validation dataset.
  
- **from_scratch**: *bool, default=False*
  Specifies whether to train from scratch or to download and use a pretrained backbone.
  
- **silent**: *bool, default=False*
  If set to True, disables all printing to STDOUT, defaults to False.
  
- **verbose**: *bool, default=True*
  If True, enables maximum verbosity.
  
#### `RetinaFaceLearner.eval`
```python
RetinaFaceLearner.eval(self, dataset, verbose, use_subset, subset_size, pyramid, flip)
```

Performs evaluation on a dataset or a subset of a dataset.
Parameters:
- **dataset**: *object*
  Object that holds dataset to perform evaluation on.

- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
- **use_subset**: *bool, default=False*
  If True, evaluation is perform on a random subset of **subset_size** images from the **dataset**.
  
- **subset_size**: *int, default=250*
  Only used if **used_subset=True**. Specifies the number of images to perform evaluation on.
  
- **flip**: *bool, default=True*
  Specifies whether to perform image flipping during evaluation to increase performance. Increases evaluation time.
  
- **pyramid**: *bool, default=True*
  Specifies whether to use an image pyramid during evaluation to increase performance. Increases evaluation time.
  
#### `RetinaFaceLearner.infer`
```python
RetinaFaceLearner.infer(self, img, threshold=0.8, nms_threshold=0.4, scales=[1024, 1980], mask_thresh=0.8)
```

Performs inference on a single image.
Parameters:
- **img**: *object*
  Object of type engine.data.Image.
    
- **threshold**: *float, default=0.8*
  Defines the detection threshold. Bounding boxes with confidence under this value are discarded.
  
- **nms_threshold**: *float, default=0.45*
  Defines the NMS threshold.
  
- **scales**: *list, default=[1024, 1980]*
  Defines the image scales to perform detection on.
  
- **mask_thresh**: *float, default=0.8*
  If using an *'mnet'* model which can detect masked faces, defines the mask detection threshold. Bounding boxes with mask 
  confidence under this value are deemed to be maskless faces.
  
#### `RetinaFaceLearner.save`
```python
RetinaFaceLearner.save(self, path, verbose)
```

Saves a model in OpenDR format at the specified path. The model name is extracted from the base folder in the specified path.
Parameters:
- **path**: *str*
  Specifies the folder where the model will be saved. The model name is extracted from the base folder of this path.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `RetinaFaceLearner.load`
```python
RetinaFaceLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:
- **path**: *str*
  Specifies the folder where the model will be loaded from.
  
- **verbose**: *bool default=True*
  If True, enables maximum verbosity.
  
#### `RetinaFaceLearner.download`
```python
RetinaFaceLearner.download(self, path, mode, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., data for training, pretrained models and backbones as 
well as test data.
Parameters:
- **path**: *str, default=None*
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
  
- **mode**: *{'pretrained', 'images', 'backbone', 'annotations'}, default='pretrained'*
  If *'pretrained'*, downloads a pretrained detector model based on *self.backbone*. If *'backbone'*, downloads a pretrained 
  backbone network based on *self.backbone*, used to train a model from pretrained weights. If *'images'*, downloads an 
  image to perform inference on. If *'annotations'*, downloads additional landmark annotations to be used during training.
  
- **verbose**: *bool, default=True*
  If True, maximum verbosity if enabled.
  
- **url**: *str, default=OpenDR FTP URL* 
  URL of the FTP server.
  
#### References
<a name="retinaface-1" href="https://arxiv.org/abs/1905.00641">[1]</a> RetinaFace: Single-stage Dense Face Localisation in the Wild,
[arXiv](https://arxiv.org/abs/1905.00641).