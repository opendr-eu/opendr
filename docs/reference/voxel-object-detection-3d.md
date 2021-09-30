## voxel_object_detection_3d module

The *voxel_object_detection_3d* module contains the *VoxelObjectDetection3DLearner* class, which inherits from the abstract class *Learner*.

### Class VoxelObjectDetection3DLearner
Bases: `engine.learners.Learner`

The *VoxelObjectDetection3DLearner* class is a wrapper of the TANet[[1]](#object-detectiond-3d-1) and PointPillars[[2]](#object-detectiond-3d-2) implementation found on [happinesslz/TANet](
https://github.com/happinesslz/TANet)[[3]](#object-detectiond-3d-3).
It can be used to perform voxel-based 3d object detection on point clouds and train new models.

The [VoxelObjectDetection3DLearner](#src.opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner.py) class has the
following public methods:

#### `VoxelObjectDetection3DLearner` constructor
```python
VoxelObjectDetection3DLearner(self, model_config_path, lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, scale, tanet_config_path, optimizer_params, lr_schedule_params)
```

Constructor parameters:
- **model_config_path**: *string*
  Specifies the path to the [proto](#proto) file describing the model structure and the training procedure.
- **lr**: *float, default=0.0002*  
  Specifies the initial learning rate to be used during training.
- **optimizer**: *str {'rms_prop_optimizer', 'momentum_optimizer', 'adam_optimizer'}, default=adam_optimizer*  
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str {'constant_learning_rate', 'exponential_decay_learning_rate', 'manual_step_learning_rate', 'cosine_decay_learning_rate'}, default='exponential_decay_learning_rate'*  
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda', 'cuda:x'}, default='cuda:0'*  
  Specifies the device to be used.
- **tanet_config_path**: *str, default=None*  
  Specifies if the configuration file for TANet should be different from the standard one (`None` for standard).
- **optimizer_params**: *dict, default={
      "weight_decay": 0.0001,
  }*  
  Specifies the parameters for the used optimizer.
    - `adam_optimizer` uses "weight_decay" values
    - `momentum_optimizer` uses "momentum_optimizer_value", "weight_decay" values
    - `rms_prop_optimizer` uses "decay", "momentum_optimizer_value", "epsilon", "weight_decay" values
- **iters**: *int, default=10*  
  Skipped. The number of training iterations is described in a [proto](#proto) file.
- **batch_size**: *int, default=64*
  Skipped. The batch size is described in a [proto](#proto) file.
- **backbone**: *str, default='tanet_16'*  
  Skipped. The structure of a model is described in a [proto](#proto) file.
- **network_head**: *str, default='tanet_16'*  
  Skipped. The structure of a model is described in a [proto](#proto) file.
- **threshold**: *float, default=0.0*  
  Skipped. The structure of a model is described in a [proto](#proto) file.
- **scale**: *float, default=1.0*  
  Skipped. The structure of a model is described in a [proto](#proto) file.


#### `VoxelObjectDetection3DLearner.fit`
```python
VoxelObjectDetection3DLearner.fit(self, dataset, val_dataset, refine_weight, ground_truth_annotations, logging_path, silent, verbose, model_dir, image_shape, evaluate)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Parameters:
  - **dataset**: *object*  
    Object that holds the training dataset.
    Can be of type `ExternalDataset` (with type="kitti") or a custom dataset inheriting from `DatasetIterator`.
  - **val_dataset**: *object, default=None*
    Object that holds the validation dataset. If None, and the dataset is an `ExternalDataset`, dataset will be used to sample evaluation inputs. Can be of type `ExternalDataset` (with type="kitti") or a custom dataset inheriting from `DatasetIterator`.
  - **refine_weight**: *float, default=2***  
    Defines the weight for the refinement part in the loss (for TANet).
  - **ground_truth_annotations**: *list of BoundingBox3DList, default=None*  
    Can be used to provide modified ground truth annotations.
  - **logging_path**: *str, default=None*  
    Path to save log files. If set to None, only the console will be used for logging.
  - **silent**: *bool, default=False*  
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=False*  
    If set to True, enables maximum verbosity.
  - **model_dir**: *str, default=None***  
    Can be used for storing and loading checkpoints.
  - **image_shape**: *(int, int), default=(1224, 370)***  
    Camera image shape for KITTI evaluation.
  - **evaluate**: *str, default=True*  
    Should the evaluation be run during training.

#### `VoxelObjectDetection3DLearner.eval`
```python
VoxelObjectDetection3DLearner.eval(self, dataset, ground_truth_annotations, logging_path, silent, verbose, image_shape, count)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
- **dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **ground_truth_annotations**: *list of BoundingBox3DList, default=None*  
  Can be used to provide modified ground truth annotations.
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=False*  
  If set to True, enables the maximum verbosity.
- **image_shape**: *(int, int), default=(1224, 370)***  
  Camera image shape for KITTI evaluation.
- **count**: *int, default=None***  
  Specifies the number of frames to be used for evaluation. If None, the full dataset is used.

#### `VoxelObjectDetection3DLearner.infer`
```python
VoxelObjectDetection3DLearner.infer(self, point_clouds)
```

This method is used to perform 3D object detection on a point cloud.
Returns a list of [BoundingBox3DList](#class_engine.target.BoundingBox3DList) objects if the list of [PointCloud](#class_engine.data.PointCloud) is given or a single [BoundingBox3DList](#class_engine.target.BoundingBox3DList) if a single [PointCloud](#class_engine.data.PointCloud) is given.

Parameters:
- **point_clouds**: *engine.data.PointCloud* or *[engine.data.PointCloud]***  
  Input data.

#### `VoxelObjectDetection3DLearner.save`
```python
VoxelObjectDetection3DLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as "name_vfe.pth", "name_mfe.pth", and "name_rpn.pth" or "name_rpn.onnx" and the metadata file as "name.json".
If the directory already exists, the files are overwritten.

If [`self.optimize`](#VoxelObjectDetection3DLearner.optimize) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the `self.temp_path` it was saved previously during conversion.

Parameters:
- **path**: *str*  
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `VoxelObjectDetection3DLearner.load`
```python
VoxelObjectDetection3DLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `VoxelObjectDetection3DLearner.optimize`
```python
VoxelObjectDetection3DLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.

#### `VoxelObjectDetection3DLearner.download`
```python
@staticmethod
VoxelObjectDetection3DLearner.download(model_name, path, server_url)
```

Download utility for pretrained models.

Parameters:
- **model_name**: *str {'pointpillars_car_xyres_16', 'pointpillars_ped_cycle_xyres_16', 'tanet_car_xyres_16', 'tanet_ped_cycle_xyres_16'}*
  The name of the model to download.
- **path**: *str*
  Local path to save the downloaded files.
- **server_url**: *str, default=None*  
  URL of the pretrained models directory on an FTP server. If None, OpenDR FTP URL is used.


#### Examples

* **Training example using an `ExternalDataset`**.  
  Mini and nano KITTI dataset can be downloaded from OpenDR server.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from opendr.perception.object_detection_3d.datasets.kitti import KittiDataset

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "opendr", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "opendr", "perception", "object_detection_3d",
    "datasets", "nano_kitti_subsets")

  dataset = KittiDataset.download_nano_kitti(
      temp_dir, True, subsets_path
  )

  learner = VoxelObjectDetection3DLearner(
      model_config_path=config, device=DEVICE,
      checkpoint_after_iter=2,
  )

  learner.fit(
      dataset,
      model_dir=model_path,
      verbose=True,
      evaluate=False,
  )
  learner.save(model_path)
  ```

* **Training example using a `DatasetIterator`**.  
  If the DatasetIterator is given as a dataset, `val_dataset` should be specified.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from opendr.perception.object_detection_3d.datasets.kitti import LabeledPointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "opendr", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "opendr", "perception", "object_detection_3d",
    "datasets", "nano_kitti_subsets")

  dataset_path = KittiDataset.download_nano_kitti(
      temp_dir, True, subsets_path
  ).path

  dataset = LabeledPointCloudsDatasetIterator(
    dataset_path + "/training/velodyne_reduced",
    dataset_path + "/training/label_2",
    dataset_path + "/training/calib",
  )

  val_dataset = LabeledPointCloudsDatasetIterator(
    dataset_path + "/training/velodyne_reduced",
    dataset_path + "/training/label_2",
    dataset_path + "/training/calib",
  )

  learner = VoxelObjectDetection3DLearner(
    model_config_path=config, device=DEVICE,
    checkpoint_after_iter=90,
  )

  learner.fit(
    dataset,
    val_dataset=val_dataset,
    model_dir=model_path,
  )
  learner.save(model_path)
  ```

* **Inference example.**
  ```python
  import os
  import torch
  from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from opendr.engine.datasets import PointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "opendr", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "opendr", "perception", "object_detection_3d",
    "datasets", "nano_kitti_subsets")

  dataset_path = KittiDataset.download_nano_kitti(
      temp_dir, True, subsets_path
  ).path

  dataset = PointCloudsDatasetIterator(dataset_path + "/testing/velodyne_reduced")

  learner = VoxelObjectDetection3DLearner(
      model_config_path=config, device=DEVICE
  )

  result = learner.infer(
      dataset[0]
  )

  print(result)

  result = learner.infer(
      [dataset[0], dataset[1], dataset[2]]
  )

  print(result)
  ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running `self.optimize`.
  ```python
  import os
  import torch
  from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from opendr.engine.datasets import PointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "opendr", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "opendr", "perception", "object_detection_3d",
    "datasets", "nano_kitti_subsets")

  dataset_path = KittiDataset.download_nano_kitti(
      temp_dir, True, subsets_path
  ).path

  dataset = PointCloudsDatasetIterator(dataset_path + "/testing/velodyne_reduced")

  learner = VoxelObjectDetection3DLearner(
      model_config_path=config, device=DEVICE,
      temp_path=temp_dir
  )
  learner.optimize()

  result = learner.infer(
      dataset[0]
  )

  print(result)
  ```

### <a name="proto">Proto structure</a>

Proto files can be found in [voxel_object_detection_3d/second_detector/configs](#src.opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector_configs)
- **model**:
  Specifies the model architecture under the "second" block.
  - **voxel_generator**:
    Specifies the parameters of the voxel generation.
      - **point_cloud_range**:
        Specifies the limit range of the points to be used in the [minx, miny, minz, maxx, maxy, maxz] format in meters.
      - **voxel_size**:
        Specifies the size of each voxel in the [x, y, z] format in meters.
      - **max_number_of_points_per_voxel**:
        Specifies the maximum number of points to be used in one voxel.
  - **num_class**:
    Specifies the number of classes to detect.
  - **voxel_feature_extractor**:
    Specifies the name and parameters of the voxel feature extractor layer that generates voxel-wise features.
  - **middle_feature_extractor**:
    Specifies the name and parameters of the middle feature extractor layer that creates 2D pseudo-image.
  - **rpn**:
    Specifies the name and parameters of the main layer that uses 2D pseudo-image to generate predictions.
  - **loss**:
    Specifies the loss function and its parameters.
  - **use_sigmoid_score**:
    Specifies if the sigmoid function should be used for score.
  - **encode_background_as_zeros**:
    Specifies if the background class should be encoded as zeros or as a separate class.
  - **encode_rad_error_by_sin**:
    Specifies if the sin function should be used for rad error.
  - **use_direction_classifier**:
    Specifies if a separate branch should be created to classify direction of the object.
  - **direction_loss_weight**:
    Specifies the loss weight for direction classifier.
  - **pos_class_weight**:
    Specifies the loss weight for positive classes.
  - **neg_class_weight**:
    Specifies the loss weight for negative classes.
  - **loss_norm_type**:
    Specifies the loss normalization type.
  - **post_center_limit_range**:
    Specifies the limit range of the predicted object centers in [minx, miny, minz, maxx, maxy, maxz] format in meters.
  - **use_rotate_nms**:
    Specifies if the rotate_nms function should be used for non-max suppression.
  - **use_multi_class_nms**:
    Specifies if the multi_class_nms function should be used for non-max suppression.
  - **use_bev**:
    Specifies if the Birds Eye View (BEV) data should be used in RPN.
  - **box_coder**:
    Specifies the box encoding strategy.
  - **target_assigner**:
    Specifies the target generator.
    - **anchor_generators**:
      Specifies the strategy for anchor generation.
      - **sizes**:
        Specifies anchor sizes in [w, l, h] format in meters.
      - **strides**:
        Specifies anchor strides in [x, y, z] format in meters.
      - **offsets**:
        Specifies anchor offsets in [x, y, z] format in meters.
      - **rotations**:
        Specifies anchor rotation range in [min, max] format in radiances.
- **train_input_reader**:
  Specifies tha data generation process for training.
  - **record_file_path**:
    Specifies the relative file path for kitti_train.tfrecord generated during data preprocessing.
  - **class_names**:
    Specifies the class names that should be used in the training of a current model.
  - **max_num_epochs**:
    Specifies the max number of epochs for training.
  - **batch_size**:
    Specifies the batch size.
  - **prefetch_size**:
    Specifies the number of prefetched data.
  - **max_number_of_voxels**:
    Specifies the max number of voxels that can be generated for one point cloud.
  - **shuffle_points**:
    Specifies if the points should be shuffled.
  - **num_workers**:
    Specifies the number of workers for the data loader.
  - **groundtruth_localization_noise_std**:
    Specifies the noise std for ground truth location.
  - **groundtruth_rotation_uniform_noise**:
    Specifies the noise range for ground truth rotation.
  - **global_scaling_uniform_noise**:
    Specifies the noise range for ground truth scale.
  - **groundtruth_points_drop_percentage**:
    Specifies the percentage of points that can be discarded.
  - **database_sampler**:
    Specifies the sapling strategy from the dataset

- **train_config**:
  Specifies the training procedure parameters.
  - **steps**:
    Specifies the number of steps to train the model.
  - **steps_per_eval**:
    Specifies the number of steps for evaluation.
- **eval_input_reader**:
  Specifies tha data generation process for evaluation.
  - **record_file_path**:
    Specifies the relative file path for kitti_val.tfrecord generated during data preprocessing.
  - **class_names**:
    Specifies the class names that should be used in the training of a current model.
  - **max_num_epochs**:
    Specifies the max number of epochs for training.
  - **batch_size**:
    Specifies the batch size.
  - **prefetch_size**:
    Specifies the number of prefetched data.
  - **max_number_of_voxels**:
    Specifies the max number of voxels that can be generated for one point cloud.
  - **shuffle_points**:
    Specifies if the points should be shuffled.
  - **num_workers**:
    Specifies the number of workers for the data loader.



#### References
<a name="#object-detectiond-3d-1" href="https://arxiv.org/pdf/1912.05163.pdf">[1]</a> TANet: Robust 3D Object Detection from Point Clouds with Triple Attention,
[arXiv](https://arxiv.org/pdf/1912.05163.pdf).  
<a name="#object-detectiond-3d-2" href="https://arxiv.org/abs/1812.05784">[2]</a> PointPillars: Fast Encoders for Object Detection from Point Clouds,
[arXiv](https://arxiv.org/abs/1812.05784).  
<a name="#object-detectiond-3d-3" href="https://github.com/happinesslz/TANet">[3]</a> Github: [TANet](https://github.com/happinesslz/TANet)
