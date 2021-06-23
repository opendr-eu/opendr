## efficient_ps module

Panoptic segmentation combines both semantic segmentation and instance segmentation in a single task.
While distinct foreground objects, e.g., cars or pedestrians, receive instance-wise segmentation masks, background classes such as buildings or road surface are combined in class-wide labels. 

For this task, EfficientPS has been included in the OpenDR toolkit.
The model architecture leverages a shared backbone for efficient encoding and fusing of semantically rich multi-scale features.
Two separate network heads create predictions for semantic and instance segmentation, respectively.
The final panoptic fusion model combines the output of the task-specific heads into a single panoptic segmentation map.

Website: http://panoptic.cs.uni-freiburg.de <br>
Arxiv: https://arxiv.org/abs/2004.02307 <br>
Original GitHub repository: https://github.com/DeepSceneSeg/EfficientPS

### Class EfficientPsLearner
Bases: `engine.learners.Learner`

The *EfficientPsLearner* class is a wrapper around the EfficientPS implementation of the original author's repository adding the OpenDR interface.

The [EfficientPsLearner](#src.opendr.perception.panoptic_segmentation.efficient_ps.efficient_ps_learner.py) class the following public methods:
#### `EfficientPsLearner` constructor
```python
EfficientPsLearner(lr, iters, batch_size, optimizer, lr_schedule, momentum, weight_decay, optimizer_config, checkpoint_after_iter, temp-path, device, num_workers, seed, config_file)
```

Constructor parameters *(for default values, check the source code)*:
- **lr**: *float*
  Specifies the learning rate used during training.
- **iters**: *int*
  Specifies the number of iterations used during training.
- **batch_size**: *int*
  Specifies the size of the batches used during both training and evaluation.
- **optimizer**: *str*
  Which optimizer to use for training.
- **lr_schedule**: *dict*
  Contains additional parameters related to the learning rate.
- **momentum**: *float*
  Specifies the momentum used by the optimizer.
- **weight_decay**: *float*
  Specifies the weight decay used by the optimizer.
- **optimizer_config**: *dict*
  Contains additional parameters related to the optimizer.
- **checkpoint_after_iter**: *int*
  Specifies the interval in epochs to save checkpoints during training.
- **temp_path**: *str*
  Path to a temporary folder that will be created to evaluate the model.
- **device**: *str*
  Specifies the device to deploy the model.
- **num_workers**: *int*
  Specifies the number of workers used by the data loaders.
- **seed**: *float*
  Specifies the seed to shuffle the data during training.
- **config_file**: *str*
  Path to the config file that contains the model architecture and the data loading pipelines.

#### `EfficientPsLearner.fit`
```python
EfficientPsLearner.fit(dataset, val_dataset=None, logging_path='./work_dir', silent=False)
```

Parameters:
- **dataset**: *CityscapesDataset*, *KittiDataset*
  Specifies the dataset used to train the model.
- **val_dataset**: *CityscapesDataset*, *KittiDataset*
  If given, this dataset will be used to evaluate the current model after each epoch.
- **logging_path**: *str*
  Path to store the logging files, e.g., training progress and tensorboard logs.
- **silent**: *bool*
  If True, disables printing the training progress reports to STDOUT. The validation will still be shown.

#### `EfficientPsLearner.eval`
```python
EfficientPsLearner.eval(dataset, print_results=False)
```

Parameters:
- **dataset**: *CityscapesDataset*, *KittiDataset*
  Specifies the dataset used to evaluate the model.
- **print_results**: *bool*
  If True, the evaluation results will be formatted and printed to STDOUT.

Return:
- **evaluation_results**: *dict*
  Contains the panoptic quality (PQ), segmentation quality (SQ), and recognition quality (RQ).

#### `EfficientPsLearner.infer`
```python
EfficientPsLearner.infer(batch, return_raw_logits=False)
```

Parameters:
- **batch**: *Image*, *List[Image]*
  Image(s) to feed to the network.
- **return_raw_logits**: *bool*
  If True, the raw network output will be returned. Otherwise, the returned object will hold Tuples of Heatmaps of the OpenDR interface.
  
Return:
- **results**: *Tuple[Heatmap, Heatmap]*, *List[Tuple[Heatmap, Heatmap]]*, *numpy array*
  If return_raw_logits is False, the predicted instance and semantic segmentation maps will be returned. Otherwise, the raw network output.
  
#### `EfficientPsLearner.save`
```python
EfficientPsLearner.save(path)
```

Parameters:
- **path**: *str*
  Specifies the location to save the current model weights.
  
Return:
- **successful**: *bool*
  Returns True if the weights could be saved. Otherwise, returns False.
  
#### `EfficientPsLearner.load`
```python
EfficientPsLearner.load(path)
```

Parameters:
- **path**: *str*
  Specifies the location to load model weights.
  
Return:
- **successful**: *bool*
  Returns True if the weights could be loaded. Otherwise, returns False.
  
#### `EfficientPsLearner.download`
```python
EfficientPsLearner.download(path, mode='model', trained_on='cityscapes')
```

Parameters:
- **path**: *str*
  Specifies the location to save the downloaded file.
- **mode**: *str*
  Valid options are 'model' to download pre-trained model weights and 'test_data' to run the unit tests.
- **trained_on**: *str*
  Specifies which model weights to download. Pre-trained models are available for the Cityscapes and KITTI panoptic segmentation datasets.\
  
Return:
- **filename**: *str*
  Absolute path to the downloaded file or directory.
