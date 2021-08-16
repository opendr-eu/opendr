## semantic_segmentation module

The *semantic segmentation* module contains the *BisenetLearner* class, which inherit from the abstract class *Learner*.


### Class BisenetLearner
Bases: `engine.learners.Learner`

The *BisenetLearner* class is a wrapper of the BiseNet model [[1]](#bisenetp) found on [BiseNet] (https://github.com/ooooverflow/BiSeNet).
It is used to train Semantic Segmentation models on RGB images and run inference.





The [BisenetLearner](#src.opendr.perception.semantic_segmentation.bisenet.bisenet_learner.py) class has the
following public methods:

#### `BisenetLearner` constructor
```python
BisenetLearner(self, lr, iters, batch_size, optimizer, checkpoint_after_iter, checkpoint_load_iter, val_after, temp_path, device, loss, weight_decay, momentum, drop_last, pin_memory, num_workers, num_classes)
```

Constructor parameters:
  - **lr**: *float, default=0.01*  
    Learning rate during optimization. 
  - **iters**: *int, default=500*  
    Number of epochs to train for. 
  - **batch_size**: *int, default=2*  
    Dataloader batch size. Defaults to 2.
  - **optimizer**: *str, default="sgd"*  
    Name of optimizer to use ("sgd" ,"rmsprop", or "adam"). 
  - **checkpoint_after_iter**: *int, default=0*  
    Save chackpoint after specific epochs. 
  - **checkpoint_load_iter**: *int, default=0*  
    Unused parameter. 
  - **val_after**: *int, default=0*  
    Perform validation after specific epochs. 
  - **temp_path**: *str, default='./temp'*  
    Path in which to store temporary files. 
  - **device**: *str, default="cuda"*  
    Name of computational device ("cpu" or "cuda"). 
  - **loss_func**: *str, default=torch.nn.CrossEntropyLoss()*  
    Loss function used during optimization. 
  - **weight_decay**: *[type], default=1e-5*  
    Weight decay used for optimization. 
  - **momentum**: *float, default=0.9*  
    Momentum used for optimization. 
  - **drop_last**: *bool, default=True*  
    Drop last data point if a batch cannot be filled. 
  - **pin_memory**: *bool, default=False*  
    Pin memory in dataloader. 
  - **num_workers**: *int, default=4*  
    Number of workers in dataloader. 
  - **num_classes**: *int, default=12*  
    Number of classes to predict among. 
  - **crop_height**: *int, default=720*
    Input image height.
  - **crop_width**: *int, default=960*
    Input image width.
  - **context_path**: *str, default='resnet18'*
    Context path for the bisenet model.


#### `BisenetLearner.fit`
```python
BisenetLearner.fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=True)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:
  - **dataset**: *Dataset*:  
    Training dataset.
  - **val_dataset**: *Dataset, default=None*  
    Validation dataset. If none is given, validation steps are skipped.
  - **logging_path**: *str default=''* 
  - **silent**: *bool default=False* 
  - **verbose**: *bool default=True*  


#### `BisenetLearner.eval`
```python
BisenetLearner.eval(self, dataset, silent=False, verbose=True)
```
This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
  - **dataset**: *Dataset*  
    Dataset on which to evaluate model.
  - **silent**: *bool, default=False* 

  - **verbose**: *bool default=True*  




#### `BisenetLearner.infer`
```python
BisenetLearner.infer(img, spath=None)
```

This method is used to perform segmentation on an image
Returns a `engine.target.Heatmap` object.

Parameters:
  - **img**: *Image*  
    Image to predict a heatmap.
  - **spath**: *str*  
    Path to save an additional output as image.  


#### `BisenetLearner.save`
```python
BisenetLearner.save(self, path)
```

Save model weights and metadata to path.

Parameters:
- **path**: *str*  
  Directory in which to save model weights and meta data.


#### `BisenetLearner.load`
```python
BisenetLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.


Parameters:
- **path**: *str*  
  Path to metadata file in json format or to weights path.



#### `BisenetLearner.download`
```python
BisenetLearner.download(self, path, mode)
```

Download pretrained models and testing images to path.

Parameters:
- **path**: *str*  
  Local path to save the files.


#### References
<a name="bisenetp" href="https://arxiv.org/abs/1808.00897">[1]</a> BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,
[arXiv]https://arxiv.org/abs/1808.00897).  
