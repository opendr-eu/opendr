## object_tracking_2d_fair_mot module

The *object_tracking_2d_fair_mot* module contains the *ObjectTracking2DFairMotLearner* class, which inherits from the abstract class *Learner*.

### Class ObjectTracking2DFairMotLearner
Bases: `engine.learners.Learner`

The *ObjectTracking2DFairMotLearner* class is a wrapper of the FairMOT[[1]](#object-tracking-2d-1) implementation found on [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)[[2]](#object-tracking-2d-2).
It can be used to perform 2D object tracking on images and train new models.

The [ObjectTracking2DFairMotLearner](#src.opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner.py) class has the
following public methods:

#### `ObjectTracking2DFairMotLearner` constructor
```python
ObjectTracking2DFairMotLearner(self, lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, scale, lr_step, head_conv, ltrb, num_classes, reg_offset, gpus, num_workers, mse_loss, reg_loss, dense_wh, cat_spec_wh, reid_dim, norm_wh, wh_weight, off_weight, id_weight, num_epochs, hm_weight, down_ratio, max_objs, track_buffer, image_mean, image_std, frame_rate, min_box_area)
```

Constructor parameters:
- **lr**: *float, default=0.0001*  
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=-1*  
  Specifies the number if iteration per each training epoch. -1 means full epoch.
- **batch_size**: *int, default=4*
  Specifies the size of the training batch.
- **optimizer**: *str {'adam'}, default=adam*  
  Specifies the optimizer type that should be used.
- **backbone**: *str {'dlav0_X', 'dla_X', 'dlaconv_X', 'resdcn_X', 'resfpndcn_X', 'hrnet_X'}, default='dla_34'*  
  Specifies the structure of the network that should be used in a `name_X` format where `name` is a type of the network and `X` is a number of layers.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the onnx optimized model and checkpoints (if needed).
- **device**: *{'cpu', 'cuda', 'cuda:x'}, default='cuda'*  
  Specifies the device to be used.
- **threshold**: *float, default=0.3*  
  Specifies the confidence threshold for tracking.
- **lr_step**: *list of int, default=[20]*
  Specifies the number of epochs to change the learning step.
- **head_conv**: *int, default=256*
  Specifies the number of channels for the network head.
- **ltrb**: *bool, default=True*
  Specifies if a bounding box should be regressed as `(left, top, right, bottom)`, or as `(size, center_offset)`.
- **num_classes**: *int, default=1*
  Specifies the number of classes to track.
- **reg_offset**: *bool, default=True*
  Specifies if center offset should be regressed.
- **gpus**: *list of int, default=[0]*
  Specifies the list of gpus to use for training. Input batch is split evenly between the gpus during training.
- **num_workers**: *int, default=4*
  Specifies the number of workers for data loaders.
- **mse_loss**: *float, default=False*
  Specifies if the Mean Squared Error (MSE) loss should be used instead of the Focal loss.
- **reg_loss**: *str {'sl1', 'l1'}, default='l1'*
  Specifies the regression loss type.
- **dense_wh**: *bool, default=False*
  Specifies if weighted regression near the center point should be applied or just the regression on the center point.
- **cat_spec_wh**: *bool, default=False*
  Specifies if category specific bounding box size should be used.
- **reid_dim**: *int, default=128*
  Specifies the number of re-identification features per object. These features are used to distinguish between different objects across the sequence frames.
- **norm_wh**: *float, default=False*
  Specifies if the regression loss should be normalized.
- **wh_weight**: *float, default=0.1*
  Specifies the loss weight for bounding box size.
- **off_weight**: *float, default=1*
  Specifies the loss weight for keypoint local offsets.
- **id_weight**: *float, default=1*
  Specifies the loss weight for re-identification features. These features are used to distinguish between different objects across the sequence frames.
- **num_epochs**: *float, default=30*
  Specifies the number of epochs to train.
- **hm_weight**: *float, default=1*
  Specifies the loss weight for keypoint heatmaps.
- **down_ratio**: *float {4}, default=4*
  Specifies the output stride. Currently only supports 4.
- **max_objs**: *float, default=500*
  Specifies the max number of output objects.
- **track_buffer**: *float, default=30*
  Specifies the size of the tracking buffer.
- **image_mean**: *list of float, default=[0.408, 0.447, 0.47]*
  Specifies the mean value for input images
- **image_std**: *list of float, default=[0.289, 0.274, 0.278]*
  Specifies the standard deviation value for input images.
- **frame_rate**: *int, default=30*
  Specifies the framerate of input images for inference.
- **min_box_area**: *float, default=100*
  Specifies the minimal box area for a positive prediction.
- **network_head**: *str {''}, default=''*  
  Skipped.
- **lr_schedule**: *str {''}, default=''*  
  Skipped.
- **scale**: *float, default=1.0*  
  Skipped.


#### `ObjectTracking2DFairMotLearner.fit`
```python
ObjectTracking2DFairMotLearner.fit(
self, dataset, val_dataset, val_epochs, logging_path, silent, verbose, train_split_paths, val_split_paths, resume_optimizer, nID)
```

This method is used for training the algorithm on the train `dataset` and validating on the `val_dataset`.
Parameters:
  - **dataset**: *object*  
    Object that holds the training dataset.
    Can be of type `ExternalDataset` (with type="kitti") or a custom dataset inheriting from `DatasetIterator`.
  - **val_dataset**: *object, default=None*
    Object that holds the validation dataset.
    If None, and the dataset is an `ExternalDataset`, dataset will be used to sample evaluation inputs.
    Can be of type `ExternalDataset` (with type="mot") or a custom dataset inheriting from `DatasetIterator`.
  - **val_epochs**: *int, default=-1*  
    Defines the number of train epochs passed to start evaluation. -1 means no evaluation.
  - **logging_path**: *str, default=None*  
    Path to save log files. If set to None, only the console will be used for logging.
  - **silent**: *bool, default=False*  
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=False*  
    If set to True, enables maximum verbosity.
  - **train_split_paths**: *dict[str, str], default=None*  
    Specifies the training splits for each sub-dataset in a `{name: splits_path}` format.
    Used if the provided `dataset` is an `ExternalDataset`.
    If None, the default path for MOT20 dataset is used.
  - **val_split_paths**: *dict[str, str], default=None*  
    Specifies the validation splits for each sub-dataset in a `{name: splits_path}` format.
    Used if the provided `val_dataset` is an `ExternalDataset`.
    If None, the `train_split_paths` is used.
  - **nID**: *int, default=None*  
    Specifies the total number of identities in the dataset. If None, value from the dataset is taken.

#### `ObjectTracking2DFairMotLearner.eval`
```python
ObjectTracking2DFairMotLearner.eval(self, dataset, val_split_paths, logging_path, silent, verbose)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
- **dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_split_paths**: *dict[str, str], default=None*  
  Specifies the validation splits for each sub-dataset in a `{name: splits_path}` format. Used if the provided `val_dataset` is an `ExternalDataset`.
- **logging_path**: *str, default=None*  
  Path to save log files. If set to None, only the console will be used for logging.
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=False*  
  If set to True, enables the maximum verbosity.

#### `ObjectTracking2DFairMotLearner.infer`
```python
ObjectTracking2DFairMotLearner.infer(self, batch, frame_ids, img_size)
```

This method is used to 2d object tracking on an image.
Returns a list of `engine.target.TrackingAnnotationList` objects if the list of `engine.data.Image` is given or a single `engine.target.TrackingAnnotationList` if a single `engine.data.Image` is given.

Parameters:
- **batch**: *engine.data.Image* or a *list of engine.data.Image*  
  Input data.
- **frame_ids**: *list of int, default=None*  
  Specifies frame ids for each input image to associate output tracking boxes. If None, -1 is used for each output box.
- **img_size**: *tuple of ints, default=(1088, 608)*  
  Specifies the pre-processed images size.

#### `ObjectTracking2DFairMotLearner.save`
```python
ObjectTracking2DFairMotLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as "name.pth" or "name.onnx" and the metadata file as "name.json".
If the directory already exists, the files are overwritten.

If [`self.optimize`](#ObjectTracking2DFairMotLearner.optimize) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the `self.temp_path` it was saved previously during conversion.

Parameters:
- **path**: *str*  
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `ObjectTracking2DFairMotLearner.load`
```python
ObjectTracking2DFairMotLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `ObjectTracking2DFairMotLearner.optimize`
```python
ObjectTracking2DFairMotLearner.optimize(self, do_constant_folding, img_size, optimizable_dcn_v2)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.
- **img_size**: *tuple(int, int), default=(1088, 608)*
  The size of model input image (after data preprocessing).
- **optimizable_dcn_v2**: *bool, default=False*
  Should be set to `True` of DCNv2 (deformable convolutions) library can be optimized. Optimization of the model is impossible without the optimization of DCNv2.


#### `ObjectTracking2DFairMotLearner.download`
```python
@staticmethod
ObjectTracking2DFairMotLearner.download(model_name, path, server_url)
```

Download utility for pretrained models.

Parameters:
- **model_name**: *str {'crowdhuman_dla34', 'fairmot_dla34'}*
  The name of the model to download.
- **path**: *str*
  Local path to save the downloaded files.
- **server_url**: *str, default=None*  
  URL of the pretrained models directory on an FTP server. If None, OpenDR FTP URL is used.


#### Examples

* **Training example using an `ExternalDataset`**.  
  Nano MOT dataset can be downloaded from the OpenDR server.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
      ObjectTracking2DFairMotLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.mot_dataset import MotDataset

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "fairmot_dla34"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)
  train_split_paths = {
    "nano_mot20": os.path.join(
      ".", "src", "opendr", "perception", "object_tracking_2d",
      "datasets", "splits", "nano_mot20.train"
    )
  }

  dataset = MotDataset.download_nano_mot20(
    temp_dir, True
  )

  learner = ObjectTracking2DFairMotLearner(
    iters=3,
    num_epochs=1,
    checkpoint_after_iter=3,
    temp_path=temp_dir,
    device=DEVICE,
  )

  learner.fit(
    dataset,
    val_epochs=1,
    train_split_paths=train_split_paths,
    val_split_paths=train_split_paths,
    verbose=True,
  )

  learner.save(model_path)
  ```

* **Training example using a `DatasetIterator`**.  
  If the `DatasetIterator` is given as a dataset, `val_dataset` should be specified.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
      ObjectTracking2DFairMotLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.mot_dataset import (
    MotDataset,
    MotDatasetIterator,
    RawMotDatasetIterator,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "fairmot_dla34"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)
  train_split_paths = {
    "nano_mot20": os.path.join(
      ".", "src", "opendr", "perception", "object_tracking_2d",
      "datasets", "splits", "nano_mot20.train"
    )
  }

  dataset_path = MotDataset.download_nano_mot20(
    temp_dir, True
  ).path

  dataset = MotDatasetIterator(dataset_path, train_split_paths)
  eval_dataset = RawMotDatasetIterator(dataset_path, train_split_paths)

  learner = ObjectTracking2DFairMotLearner(
    iters=3,
    num_epochs=1,
    checkpoint_after_iter=3,
    temp_path=temp_dir,
    device=DEVICE,
  )

  learner.fit(
    dataset,
    val_dataset=eval_dataset,
    val_epochs=1,
    train_split_paths=train_split_paths,
    val_split_paths=train_split_paths,
    verbose=True,
  )

  learner.save(model_path)
  ```

* **Inference example.**
  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
      ObjectTracking2DFairMotLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.mot_dataset import (
    MotDataset,
    RawMotDatasetIterator,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "fairmot_dla34"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)
  train_split_paths = {
    "nano_mot20": os.path.join(
      ".", "src", "opendr", "perception", "object_tracking_2d",
      "datasets", "splits", "nano_mot20.train"
    )
  }

  dataset_path = MotDataset.download_nano_mot20(
    temp_dir, True
  ).path

  eval_dataset = RawMotDatasetIterator(dataset_path, train_split_paths)

  learner = ObjectTracking2DFairMotLearner(
    iters=3,
    num_epochs=1,
    checkpoint_after_iter=3,
    temp_path=temp_dir,
    device=DEVICE,
  )

  result = learner.infer([
      eval_dataset[0][0],
      eval_dataset[1][0],
  ], [26, 5])

  print(result)
  ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running `self.optimize`.
  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
      ObjectTracking2DFairMotLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.mot_dataset import (
    MotDataset,
    RawMotDatasetIterator,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "fairmot_dla34"
  temp_dir = "temp"

  learner = ObjectTracking2DFairMotLearner(
    temp_path=self.temp_dir,
    device=DEVICE,
  )
  learner.optimize()

  dataset_path = MotDataset.download_nano_mot20(
    temp_dir, True
  ).path

  eval_dataset = RawMotDatasetIterator(dataset_path, train_split_paths)

  result = learner.infer([
      eval_dataset[0][0],
      eval_dataset[1][0],
  ], [26, 5])

  print(result)
  ```

#### References
<a name="#object-tracking-2d-1" href="https://arxiv.org/abs/2004.01888">[1]</a> FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking,
[arXiv](https://arxiv.org/abs/2004.01888).  
<a name="#object-tracking-2d-2" href="https://github.com/ifzhang/FairMOT">[2]</a> Github: [ifzhang/FairMOT](
https://github.com/ifzhang/FairMOT)
