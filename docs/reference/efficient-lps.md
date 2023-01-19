## efficient_lps module

Panoptic segmentation combines both semantic segmentation and instance segmentation in a single task.
While distinct foreground objects, e.g., cars or pedestrians, receive instance-wise segmentation masks, background classes such as buildings or road surface are combined in class-wide labels.

For this task, EfficientLPS has been included in the OpenDR toolkit.
The model architecture leverages a shared backbone for efficient encoding and fusing of semantically rich multi-scale features.
Two separate network heads create predictions for semantic and instance segmentation, respectively.
The final panoptic fusion model combines the output of the task-specific heads into a single panoptic segmentation map.

Website: [http://lidar-panoptic.cs.uni-freiburg.de](http://lidar-panoptic.cs.uni-freiburg.de) <br>
Arxiv: [https://arxiv.org/abs/2102.08009](https://arxiv.org/abs/2102.08009) <br>
Original GitHub repository: [https://github.com/robot-learning-freiburg/EfficientLPS](https://github.com/robot-learning-freiburg/EfficientLPS)

### Class EfficientLpsLearner
Bases: `engine.learners.Learner`

The *EfficientLpsLearner* class is a wrapper around the EfficientLPS implementation of the original author's repository adding the OpenDR interface.

The [EfficientLpsLearner](/src/opendr/perception/panoptic_segmentation/efficient_lps/efficient_lps_learner.py) class has the following public methods:
#### `EfficientLpsLearner` constructor
```python
EfficientLpsLearner(config_file, lr, iters, batch_size, optimizer, lr_schedule, momentum, weight_decay, optimizer_config, checkpoint_after_iter, temp_path, device, num_workers, seed)
```

Constructor parameters:

- **config_file**: *str*\
  Path to the config file that contains the model architecture and the data loading pipelines.
- **lr**: *float, default=0.07*\
  Specifies the learning rate used during training.
- **iters**: *int, default=160*\
  Specifies the number of iterations used during training.
- **batch_size**: *int, default=1*\
  Specifies the size of the batches used during both training and evaluation.
- **optimizer**: *str, default='SGD'*\
  Which optimizer to use for training.
- **lr_schedule**: *dict[str, any], default=None*\
  Contains additional parameters related to the learning rate.
- **momentum**: *float, default=0.9*\
  Specifies the momentum used by the optimizer.
- **weight_decay**: *float, default=0.0001*\
  Specifies the weight decay used by the optimizer.
- **optimizer_config**: *dict[str, any], default=None*\
  Contains additional parameters related to the optimizer.
- **checkpoint_after_iter**: *int, default=1*\
  Specifies the interval in epochs to save checkpoints during training.
- **temp_path**: *Path*, *str, default='../eval_tmp_dir'*\
  Path to a temporary folder that will be created to evaluate the model.
- **device**: *str, default='cuda:0'*\
  Specifies the device to deploy the model.
- **num_workers**: *int, default=1*\
  Specifies the number of workers used by the data loaders.
- **seed**: *float, default=None*\
  Specifies the seed to shuffle the data during training.

#### `EfficientLpsLearner.fit`
```python
EfficientLpsLearner.fit(self, dataset, val_dataset, logging_path, silent)
```

Parameters:

- **dataset**: *object*\
  Specifies the dataset used to train the model.
  Supported datasets are SemanticKitti and NuScenes (Future) (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **val_dataset**: *object*\
  If given, this dataset will be used to evaluate the current model after each epoch.
  Supported datasets are SemanticKitti and NuScenes (Future) (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **logging_path**: *Path*, *str, default='../logging'*\
  Path to store the logging files, e.g., training progress and tensorboard logs.
- **silent**: *bool, default=True*\
  If True, disables printing the training progress reports to STDOUT.
  The validation will still be shown.
  
Return:

- **results**: *dict*\
  Dictionary with "train" and "val" keys containing the training progress (e.g. losses) and, if a val_dataset is provided, the evaluation results.
#### `EfficientLpsLearner.eval`
```python
EfficientLpsLearner.eval(self, dataset, print_results)
```

Parameters:

- **dataset**: *object*\
  Specifies the dataset used to evaluate the model.
  Supported datasets are SemanticKitti and NuScenes (Future) (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **print_results**: *bool, default=False*\
  If True, the evaluation results will be formatted and printed to STDOUT.

Return:

- **evaluation_results**: *dict*\
  Contains the panoptic quality (PQ), segmentation quality (SQ), recognition quality (RQ) and Intersection over Union (IoU).


#### `EfficientLpsLearner.pcl_to_mmdet`
```python
EfficientLpsLearner.pcl_to_mmdet(self, point_cloud, frame_id)
```

Parameters:

- **point_cloud**: *PointCloud*\
  Specifies the OpenDR PointCloud object that will be converted to a MMDetector compatible object.
- **frame_id**: *int, default=0*\
  Number of the scan frame to be used as its filename.
  Inferences will use the same filename.
  
Return:

- **results**: *dict*\
  An MMDetector compatible dictionary containing the PointCloud data and some additional metadata.

#### `EfficientLpsLearner.infer`
```python
EfficientLpsLearner.infer(self, batch, return_raw_logits, projected)
```

Parameters:

- **batch**: *PointCloud*, *List[PointCloud]*\
  Point Cloud(s) to feed to the network.
- **return_raw_logits**: *bool, default=False*\
  If True, the raw network output will be returned.
  Otherwise, the returned object will hold Tuples of Heatmaps of the OpenDR interface.
- **projected**: *bool, default=False*\
  If True, output will be returned as 2D heatmaps of the spherical projections of the semantic and instance labels, as well as the spherical projection of the scan's range.
  Otherwise, the semantic and instance labels will be returned as Numpy arrays for each point.

Return:

- **results**: *Tuple[Heatmap, Heatmap, Image]*,\
  *Tuple[np.ndarray, np.ndarray, None]*,\
  *List[Tuple[Heatmap, Heatmap, Image]]*,\
  *List[Tuple[np.ndarray, np.ndarray, None]]*\
  If *return_raw_logits* is True the raw network output will be returned.
  If *return_raw_logits* is True and *projected* is true, the predicted instance and semantic segmentation maps, as well as the scan range map will be returned.
  Otherwise, if *return_raw_logits* is True and *projected* is false, the predicted instance and semantic labels for each point will be returned as numpy arrays.

#### `EfficientLpsLearner.save`
```python
EfficientLpsLearner.save(self, path)
```

Parameters:

- **path**: *Path*, *str*\
  Specifies the location to save the current model weights.

Return:

- **successful**: *bool*\
  Returns True if the weights could be saved.
  Otherwise, returns False.

#### `EfficientLpsLearner.load`
```python
EfficientLpsLearner.load(path)
```

Parameters:

- **path**: *Path*, *str*\
  Specifies the location to load model weights.

Return:

- **successful**: *bool*\
  Returns True if the weights could be loaded.
  Otherwise, returns False.

#### `EfficientLpsLearner.download`
```python
EfficientLpsLearner.download(path, mode, trained_on)
```

Parameters:

- **path**: *str*\
  Specifies the location to save the downloaded file.
- **mode**: *str, default='model'*\
  Valid options are *'model'* to download pre-trained model weights and *'test_data'* to run the unit tests.
- **trained_on**: *str, default='semantickitti'*\
  Specifies which model weights to download.
  Pre-trained models are available for the SemanticKITTI and NuScenes (Future) datasets.

Return:

- **filename**: *str*\
  Absolute path to the downloaded file or directory.

#### `EfficientLpsLearner.visualize`
```python
EfficientLpsLearner.visualize(self, pointcloud, predictions, show_figure, save_figure, figure_filename, figure_size, max_inst, min_alpha, dpi)
```

Parameters:

- **pointcloud**: *PointCloud*\
  PointCloud used for inference.
- **prediction**: *Tuple[np.ndarray, np.ndarray]*\
  The semantic and instance segmentation labels obtained with the `infer()` method and *projected* set to False.
- **show_figure**: *bool, default=True*\
  If True, the generated figure will be shown on screen.
- **save_figure**: *bool, default=False*\
  If True, the generated figure will be saved to disk. The **figure_filename** has to be set.
- **figure_filename**: *Path*, *str, default=None*\
  The path used to save the figure if **save_figure** is True.
- **figure_size**: *Tuple[float, float], default=(15, 10)*\
  The size of the figure in inches.
- **max_inst**: *int, default=20*\
  Maximum value that the instance ID can take.
  Used for computing the alpha value of a point.
- **min_alpha**: *float, default=0.25*\
  Minimum value that a point's alpha value can take, so that it is never fully transparent.
- **dpi**: *int, default=600*\
  Resolution of the resulting image, in Dots per Inch.
- **return_pointcloud**: *Optional[bool], default=False*\
  If True, returns a PointCloud object with the predicted labels as colors.
- **return_pointcloud_type**: *Optional[str], default=None*\
  If return_pointcloud is True, this parameter specifies the type of the returned PointCloud object.
  Valid options are "semantic", "instance" and "panoptic".

Return:

- **visualization**: *Union[PointCloud, Image]*\
  OpenDR Image of the generated visualization or OpenDR PointCloud with the predicted labels.

#### Performance Evaluation

The speed of pointcloud per second is evaluated for the SemanticKITTI dataset:

| Dataset    | ~ | GeForce GTX TITAN X | ~ | Xavier AGX |
|------------|-----------------|---------------------|-----------|------------|
| SemanticKITTI | ~             | 0.53                | ~       | ~        |

The memory and energy usage is evaluated for different datasets.
An NVIDIA Jetson Xavier AGX was used as the reference platform for energy measurements.
The reported memory is the max number seen during evaluation on the respective validation set.
The energy is measured during the evaluation.

| Dataset                | Memory (MB) | Energy (Joules) - Total per inference AGX |
|------------------------|-------------|-------------------------------------------|
| SemanticKITTI | ~       | ~                                      |

The performance is evaluated using three different metrics, namely Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

| Dataset    | PQ   | SQ   | RQ   |
|------------|------|------|------|
| SemanticKITTI | 52.5 | 72.6 | 63.1 |

EfficientPS is compatible with the following platforms:

| Platform                                     | Compatibility |
|----------------------------------------------|---------------|
| x86 - Ubuntu 20.04 (bare installation - CPU) | ❌            |
| x86 - Ubuntu 20.04 (bare installation - GPU) | ✔️            |
| x86 - Ubuntu 20.04 (pip installation)        | ❌            |
| x86 - Ubuntu 20.04 (CPU docker)              | ❌            |
| x86 - Ubuntu 20.04 (GPU docker)              | ✔️            |
| NVIDIA Jetson Xavier AGX                     | ✔️            |
