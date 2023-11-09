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

The [EfficientPsLearner](/src/opendr/perception/panoptic_segmentation/efficient_ps/efficient_ps_learner.py) class has the following public methods:
#### `EfficientPsLearner` constructor
```python
EfficientPsLearner(config_file, lr, iters, batch_size, optimizer, lr_schedule, momentum, weight_decay, optimizer_config, checkpoint_after_iter, temp_path, device, num_workers, seed)
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
- **temp_path**: *str, default='../eval_tmp_dir'*\
  Path to a temporary folder that will be created to evaluate the model.
- **device**: *str, default='cuda:0'*\
  Specifies the device to deploy the model.
- **num_workers**: *int, default=1*\
  Specifies the number of workers used by the data loaders.
- **seed**: *float, default=None*\
  Specifies the seed to shuffle the data during training.

#### `EfficientPsLearner.fit`
```python
EfficientPsLearner.fit(dataset, val_dataset, logging_path, silent)
```

Parameters:

- **dataset**: *object*\
  Specifies the dataset used to train the model. Supported datasets are CityscapesDataset and KittiDataset (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **val_dataset**: *object*\
  If given, this dataset will be used to evaluate the current model after each epoch. Supported datasets are CityscapesDataset and KittiDataset (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **logging_path**: *str, default='../logging'*\
  Path to store the logging files, e.g., training progress and tensorboard logs.
- **silent**: *bool, default=False*\
  If True, disables printing the training progress reports to STDOUT. The validation will still be shown.

Return:

- **results**: *dict*\
  Dictionary with "train" and "val" keys containing the training progress (e.g. losses) and, if a val_dataset is provided, the evaluation results.

#### `EfficientPsLearner.eval`
```python
EfficientPsLearner.eval(dataset, print_results)
```

Parameters:

- **dataset**: *object*\
  Specifies the dataset used to evaluate the model. Supported datasets are CityscapesDataset and KittiDataset (see [readme](../../src/opendr/perception/panoptic_segmentation/datasets/README.md)).
- **print_results**: *bool, default=False*\
  If True, the evaluation results will be formatted and printed to STDOUT.

Return:

- **evaluation_results**: *dict*\
  Contains the panoptic quality (PQ), segmentation quality (SQ), and recognition quality (RQ).

#### `EfficientPsLearner.infer`
```python
EfficientPsLearner.infer(batch, return_raw_logits)
```

Parameters:

- **batch**: *Image*, *List[Image]*\
  Image(s) to feed to the network.
- **return_raw_logits**: *bool, default=False*\
  If True, the raw network output will be returned. Otherwise, the returned object will hold Tuples of Heatmaps of the OpenDR interface.

Return:

- **results**: *Tuple[Heatmap, Heatmap]*, *List[Tuple[Heatmap, Heatmap]]*, *numpy array*
  If return_raw_logits is False, the predicted instance and semantic segmentation maps will be returned. Otherwise, the raw network output.

#### `EfficientPsLearner.save`
```python
EfficientPsLearner.save(path)
```

Parameters:

- **path**: *str*\
  Specifies the location to save the current model weights.

Return:

- **successful**: *bool*\
  Returns True if the weights could be saved. Otherwise, returns False.

#### `EfficientPsLearner.load`
```python
EfficientPsLearner.load(path)
```

Parameters:

- **path**: *str*\
  Specifies the location to load model weights.

Return:

- **successful**: *bool*\
  Returns True if the weights could be loaded. Otherwise, returns False.

#### `EfficientPsLearner.download`
```python
EfficientPsLearner.download(path, mode, trained_on)
```

Parameters:

- **path**: *str*\
  Specifies the location to save the downloaded file.
- **mode**: *str, default='model'*\
  Valid options are *'model'* to download pre-trained model weights and *'test_data'* to run the unit tests.
- **trained_on**: *str, default='cityscapes'*\
  Specifies which model weights to download. Pre-trained models are available for the Cityscapes and KITTI panoptic segmentation datasets.

Return:

- **filename**: *str*\
  Absolute path to the downloaded file or directory.

#### `EfficientPsLearner.visualize`
```python
EfficientPsLearner.visualize(image, prediction, show_figure, save_figure, figure_filename, figure_size, detailed)
```

Parameters:

- **image**: *Image*\
  BGR image used for inference.
- **prediction**: *Tuple[Heatmap, Heatmap]*\
  The semantic and instance segmentation maps obtained with the `infer()` method.
- **show_figure**: *bool, default=True*\
  If True, the generated figure will be shown on screen.
- **save_figure**: *bool, default=False*\
  If True, the generated figure will be saved to disk. The **figure_filename** has to be set.
- **figure_filename**: *str, default=None*\
  The path used to save the figure if **save_figure** is True.
- **figure_size**: *Tuple[float, float]*\
  The size of the figure in inches. Only used for the detailed version. Otherwise, the size of the input data is used.
- **detailed**: *bool, default=False*\
  If True, the generated figure will be a compilation of the input color image, the semantic segmentation map, a contours plot showing the individual objects, and a combined panoptic segmentation overlay on the color image. Otherwise, only the latter will be shown.


#### Performance Evaluation

The speed (fps) is evaluated for the Cityscapes dataset (2048x1024 pixels):

| Dataset    | GeForce GTX 980 | GeForce GTX TITAN X | TITAN RTX | Xavier AGX |
|------------|-----------------|---------------------|-----------|------------|
| Cityscapes | 1.3             | 1.1                 | 3.2       | 1.7        |

The memory and energy usage is evaluated for different datasets.
An NVIDIA Jetson Xavier AGX was used as the reference platform for energy measurements.
Note that the exact number for the memory depends on the image resolution and the number of instances in an image.
The reported memory is the max number seen during evaluation on the respective validation set.
The energy is measured during the evaluation.

| Dataset                | Memory (MB) | Energy (Joules) - Total per inference AGX |
|------------------------|-------------|-------------------------------------------|
| Cityscapes (2048x1024) | 11812       | 39.3                                      |
| Kitti (1280x384)       | 3328        | 15.1                                      |

The performance is evaluated using three different metrics, namely Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

| Dataset    | PQ   | SQ   | RQ   |
|------------|------|------|------|
| Cityscapes | 64.4 | 81.8 | 77.7 |
| Kitti      | 42.6 | 77.2 | 53.1 |

EfficientPS is compatible with the following platforms:

| Platform                                     | Compatibility |
|----------------------------------------------|---------------|
| x86 - Ubuntu 20.04 (bare installation - CPU) | ❌            |
| x86 - Ubuntu 20.04 (bare installation - GPU) | ✔️            |
| x86 - Ubuntu 20.04 (pip installation)        | ❌            |
| x86 - Ubuntu 20.04 (CPU docker)              | ❌            |
| x86 - Ubuntu 20.04 (GPU docker)              | ✔️            |
| NVIDIA Jetson Xavier AGX                     | ✔️            |
