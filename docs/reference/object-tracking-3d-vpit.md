## object_tracking_3d_vpit module

The *object_tracking_3d_vpit* module contains the *ObjectTracking3DVpitLearner* class, which inherits from the abstract class *Learner*.

### Class ObjectTracking3DVpitLearner
Bases: `engine.learners.Learner`

The *ObjectTracking3DVpitLearner* class is an implementation of the VPIT[[1]](#object-tracking-3d-1) method.
Evaluation code is based on the KITTI evaluation development kit[[2]](#object-tracking-3d-2).
It can be used to perform 3D object tracking based on provided detections.

The [ObjectTracking3DVpitLearner](/src/opendr/perception/object_tracking_3d/single_object_tracking/vpit/vpit_object_tracking_3d_learner.py) class has the following public methods:

#### `ObjectTracking3DVpitLearner` constructor
```python
ObjectTracking3DVpitLearner(self, model_config_path, lr, optimizer, lr_schedule, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, tanet_config_path, optimizer_params, lr_schedule_params, feature_blocks, window_influence, score_upscale, rotation_penalty, rotation_step, rotations_count, rotation_interpolation, target_size, search_size, context_amount, target_feature_merge_scale, loss_function, r_pos, augment, augment_rotation, train_pseudo_image, search_type, target_type, bof_mode, bof_training_steps, extrapolation_mode, offset_interpolation, vertical_offset_interpolation, min_top_score, overwrite_strides, target_features_mode, upscaling_mode, regress_vertical_position, regression_training_isolation, vertical_regressor_type, vertical_regressor_kwargs, iters, backbone, batch_size, threshold, scale)
```

Constructor parameters:

- **model_config_path**: *string*\
  Specifies the path to the [proto](#proto) file describing the backbone model structure and training procedure.
- **lr**: *float, default=0.0001*\
  Specifies the initial learning rate to be used during training.
- **optimizer**: *str {'rms_prop_optimizer', 'momentum_optimizer', 'adam_optimizer'}, default=adam_optimizer*\
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str {'constant_learning_rate', 'exponential_decay_learning_rate', 'manual_step_learning_rate', 'cosine_decay_learning_rate'}, default='exponential_decay_learning_rate'*\
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm saves the onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda', 'cuda:x'}, default='cuda:0'*\
  Specifies the device to be used.
- **tanet_config_path**: *str, default=None*\
  Specifies if the configuration file for TANet should be different from the standard one (`None` for standard). Only used when the backbone network is TANet.
- **feature_blocks**: *int, default=1*\
  Specifies the number of feature bloks in FGN subnetwork.
- **window_influence**: *float, default=0.85*\
  Specifies the influence of a penalty window on the score map
- **score_upscale**: *float, default=8*\
  Specifies the upscaling ration for the score map
- **rotation_penalty**: *float, default=0.98*\
  Specifies the penalty for score maps with non-zero relative rotation.
- **rotation_step**: *float, default=0.15*\
  Specifies the angle difference in radians for rotation search.
- **rotations_count**: *int, default=3*\
  Specifies the number of rotation angles in the search.
- **rotation_interpolation**: *float, default=1*\
  Specifies the interpolation coefficient between the previous rotation angle and the new angle. Zero means never change the angle and one applies new angle instantly.
- **target_size**: *[int, int], default=[-1, -1]*\
  Specifies the pixel size for rescaling target object pseudo image. Value [-1, -1] results on original sizes.  
- **search_size**: *[int, int], default=[-1, -1]*\
  Specifies the pixel size for rescaling search object pseudo image. Value [-1, -1] results on original sizes.  
- **context_amount**: *float, default=0.25*\
  Specifies the amount of context added to the target region. Positive values result in a square region and negative values scale the region without chaning its aspect ratio.
- **target_feature_merge_scale**: *float, default=0.0*\
  Specifies the coefficient of merging initial target features with all consequtive features. High values result in fast merge to target features from current frame. Zero results in using initial features forever.
- **loss_function**: *{'bce', 'focal'}, default='bce'*\
  Specifies the loss function to be used.
- **r_pos**: *int, default=2*\
  Specifies the radious of positive label values around the projected target center.
- **augment**: *bool, default=False*\
  Specifies if to augment training objects with position shuffle.
- **augment_rotation**: *bool, default=True*\
  Specifies if to additionally augment training objects with rotation.
- **train_pseudo_image**: *bool, default=False*\
  Specifies if pseudo image generation should be trained.
- **search_type**: *{'normal', 'small', 'snormal', 'big', 'a+4'}, default='small'*\
  Specifies the size of the search region based on the target region size.
- **target_type**: *{'normal', 'original', 'small', 'a+4'}, default='normal'*\
  Specifies the size of the target region based on the object size.
- **bof_mode**: *object, default='none'*\
  Experimental. Specifies how to use neural bag of features. Disabled by default. 
- **bof_training_steps**: *, default=2000*\
  Experimental. Specifies the amount of training steps for neural bag of features
- **extrapolation_mode**: *{'none', 'linear', 'linear+'}, default='linear+'*\
  Specifies the extrapolation algorithm for future search region position.
- **offset_interpolation**: *float, default=0.25*\
  Specifies the interpolation speed of applying output object position.
- **vertical_offset_interpolation**: *float, default=1*\
  Specifies the interpolation speed of applying output vertical object position.
- **min_top_score**: *float, default=None*\
  Specifies the minimal value of top score to be used for object position regression.
- **overwrite_strides**: *list[int], default=None*\
  Specifies strides for the FGN subnetwork if they have to be different from the default values.
- **target_features_mode**: *{'init', 'all', 'selected', 'last'}, default='init'*\
  Specifies which target features are used to compare with the search regions.
- **upscaling_mode**: *{'none', 'raw', 'processed'}, default='none'*\
  Specifies if to use upscaling in FGN.
- **regress_vertical_position**: *bool, default=False*\
  Specifies if to regress vertical position with an additional branch.
- **regression_training_isolation**: *bool, default=False*\
  Specifies if to train only the vertical regression branch.
- **vertical_regressor_type**: *{'center_linear', 'convolutional', 'convolutional_3k'}, default='center_linear'*\
  Specifies the type of a vertical regression network.
- **vertical_regressor_kwargs**: *dict, default={}*\
  Specifies the arguments for a vertical regressor.
- **iters**: *int, default=10*\
  Skipped. The number of training iterations is described in a [proto](#proto) file.
- **batch_size**: *int, default=64*\
  Skipped. The batch size is described in a [proto](#proto) file.
- **backbone**: *str, default='pp'*\
  Skipped. The structure of a model is described in a [proto](#proto) file.
- **threshold**: *float, default=0.0*\
  Skipped. The structure of a model is described in a [proto](#proto) file.
- **scale**: *float, default=1.0*\
  Skipped. The structure of a model is described in a [proto](#proto) file.

#### `ObjectTracking3DVpitLearner.fit`
```python
ObjectTracking3DVpitLearner.fit(self, dataset, steps, val_dataset, refine_weight, ground_truth_annotations, logging_path, silent, verbose, model_dir, image_shape, evaluate, debug, load_optimizer)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

  - **dataset**: *object*\
    Object that holds the training dataset.
    Can be of type `ExternalDataset` (with type="kitti") or a custom dataset inheriting from `DatasetIterator`. If given `SiameseTrackingDatasetIterator`, or `SiameseTripletTrackingDatasetIterator`, the training procedure will be adjusted accordingly.
  - **steps**: *int, default=0***\
    How many steps to train. Zero means all the steps described in the config file.
  - **val_dataset**: *object, default=None*\
    Object that holds the validation dataset. If None, and the dataset is an `ExternalDataset`, dataset will be used to sample evaluation inputs. Can be of type `ExternalDataset` (with type="kitti") or a custom dataset inheriting from `DatasetIterator`.
  - **refine_weight**: *float, default=2*\
    Defines the weight for the refinement part in the loss (for TANet).
  - **ground_truth_annotations**: *list of BoundingBox3DList, default=None*\
    Can be used to provide modified ground truth annotations.
  - **logging_path**: *str, default=None*\
    Path to save log files. If set to None, only the console will be used for logging.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=False*\
    If set to True, enables maximum verbosity.
  - **model_dir**: *str, default=None*\
    Can be used for storing and loading checkpoints.
  - **image_shape**: *(int, int), default=(1224, 370)*\
    Camera image shape for KITTI evaluation.
  - **evaluate**: *bool, default=False*\
    Should the evaluation be run during training.
  - **debug**: *bool, default=False*\
    Should the debug plots and logs be created.
  - **load_optimizer**: *bool, default=True*\
    Should the optimizer be loaded from the checkpoint.


#### `ObjectTracking3DVpitLearner.eval`
```python
ObjectTracking3DVpitLearner.eval(self, dataset, ground_truth_annotations, logging_path, silent, verbose, image_shape, count)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:

- **dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **ground_truth_annotations**: *list of BoundingBox3DList, default=None*\
  Can be used to provide modified ground truth annotations.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=False*\
  If set to True, enables the maximum verbosity.
- **image_shape**: *(int, int), default=(1224, 370)*\
  Camera image shape for KITTI evaluation.
- **count**: *int, default=None*\
  Specifies the number of frames to be used for evaluation. If None, the full dataset is used.


#### `ObjectTracking3DVpitLearner.init`
```python
ObjectTracking3DVpitLearner.init(self, point_cloud, label_lidar, draw, clear_metrics)
```

This method is used to provide the object of interest for furhter tracking.

Parameters:

- **point_cloud**: *engine.data.PointCloud**\
  Input data.
- **label_lidar**: *engine.target.TrackingAnnotation**\
  Target object label (position, size, rotation).
- **draw**: *bool, default=False*\
  Specifies if to draw the model outputs.
- **clear_metrics**: *bool, default=False*\
  Specifies if to clear fps metrics.


#### `ObjectTracking3DVpitLearner.infer`
```python
ObjectTracking3DVpitLearner.infer(self, point_cloud, frame, id, draw)
```

This method is used to run inference on all frames after the initial frame.

Parameters:

- **point_cloud**: *engine.data.PointCloud**\
  Input data.
- **frame**: *int, default=0*\
  Frame number which is stored in the outputs of the model.
- **id**: *int, default=None*\
  Specifies if the output label id should be different from the initial id.
- **draw**: *bool, default=False*\
  Specifies if to draw the model outputs.

#### `ObjectTracking3DVpitLearner.save`
```python
ObjectTracking3DVpitLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as "name.pth" or "name.onnx" and the metadata file as "name.json".
If the directory already exists, the files are overwritten.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `ObjectTracking3DVpitLearner.load`
```python
ObjectTracking3DVpitLearner.load(self, path, verbose, backbone, full)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `ObjectTracking3DVpitLearner.download`
```python
@staticmethod
ObjectTracking3DVpitLearner.download(self, model_name, path, server_url)
```

Download utility for pretrained models.

Parameters:

- **model_name**: *str {'pointpillars_car_xyres_16', 'pointpillars_ped_cycle_xyres_16', 'tanet_car_xyres_16', 'tanet_ped_cycle_xyres_16'}*\
  The name of the backbone model to download.
- **path**: *str*\
  Local path to save the downloaded files.
- **server_url**: *str, default=None*\
  URL of the pretrained models directory on an FTP server. If None, OpenDR FTP URL is used.


#### `ObjectTracking3DVpitLearner.fps`
```python
@staticmethod
ObjectTracking3DVpitLearner.fps()
```

Returns average fps from all runs since last clear_metrics init.


#### Examples

* **Inference example.**
  ```python
  import os
  from opendr.perception.object_tracking_3d import ObjectTracking3DVpitLearner
  from opendr.perception.object_tracking_3d import (
      LabeledTrackingPointCloudsDatasetIterator,
      SiameseTrackingDatasetIterator,
  )

  DEVICE = "cpu"
  temp_dir = "temp"

  dataset = LabeledTrackingPointCloudsDatasetIterator.download_pico_kitti(
      temp_dir, True
  )

  dataset_siamese = SiameseTrackingDatasetIterator(
      [dataset.lidar_path],
      [dataset.label_path],
      [dataset.calib_path],
      classes=["Van", "Pedestrian", "Cyclist"],
  )
  learner = ObjectTracking3DVpitLearner()
  learner.init(self.dataset[0][0], self.dataset[0][2])
  result = learner.infer(self.dataset[0][1])

  print(result)

  ```

#### References
<a name="#object-tracking-3d-1" href="https://arxiv.org/abs/2008.08063">[1]</a> AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics,
[arXiv](https://arxiv.org/abs/2008.08063).
<a name="#object-tracking-3d-2" href="http://www.cvlibs.net/datasets/kitti/eval_tracking.php">[2]</a> KITTI evaluation development kit.
