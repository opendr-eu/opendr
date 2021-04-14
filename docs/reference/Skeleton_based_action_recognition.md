## skeleton_based_action_recognition module

The *skeleton_based_action_recognition* module contains the *STGCNLearner* and *PSTGCNLearner* classes, which inherits from the abstract class *Learner*.

### Class STGCNLearner
Bases: `engine.learners.Learner`

The *STGCNLearner* class is a wrapper of the ST-GCN [[1]](#1) and the proposed methods TA-GCN [[2]](#2) and ST-BLN [[3]](#3) for Skeleton-based Human 
Action Recognition.
This implementation of ST-GCN can be found in [OpenMMLAB toolbox](
https://github.com/open-mmlab/mmskeleton/tree/b4c076baa9e02e69b5876c49fa7c509866d902c7).
It can be used to perform the baseline method ST-GCN and the proposed methods TA-GCN [[2]](#2) and ST-BLN [[3]](#3) for skeleton-based action recognition. 
The TA-GCN and ST-BLN methods are proposed on top of ST-GCN and make it more efficient in terms of number of model parameters and floating point operations. 

The [STGCNLearner](#src.perception.skeleton_based_action_recognition.stgcn_learner.py) class has the
following public methods:

#### `STGCNLearner` constructor
```python
STGCNLearner(self, lr, batch_size, optimizer_name, lr_schedule,
             checkpoint_after_iter, checkpoint_load_iter, temp_path,
             device, num_workers, epochs, experiment_name,
             device_ind, val_batch_size, drop_after_epoch,
             start_epoch, dataset_name,method_name, 
             stbln_symmetric, num_frames, num_subframes=100)
```

Constructor parameters:
- **lr**: *float, default=0.01*  
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=128*  
  Specifies number of skeleton sequences to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer_name**: *str {'sgd', 'adam'}, default='sgd'*  
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str, default=' '*  
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the checkpoints and onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **num_workers**: *int, default=32*  
  Specifies the number of workers to be used by the data loader.
- **epochs**: *int, default=50*  
  Specifies the number of epochs the training should run for.
- **experiment_name**: *str, default='stgcn_nturgbd'*  
  String name to attach to checkpoints.
- **device_ind**: *list, default=[0]*  
  List of GPU indices to be used if the device is 'cuda'. 
- **val_batch_size**: *int, default=256*  
  Specifies number of skeleton sequences to be bundled up in a batch during evaluation. This heavily affects memory usage, adjust according to your system.
- **drop_after_epoch**: *list, default=[30,40]*  
  List of epoch numbers in which the optimizer drops the learning rate.  
- **start_epoch**: *int, default=0*  
  Specifies the starting epoch number for training.  
- **dataset_name**: *str {'kinetics', 'nturgbd_cv', 'nturgbd_cs'}, default='nturgbd_cv'*  
  Specifies the name of dataset that is used for training and evaluation. 
- **method_name**: *str {'stgcn', 'stbln', 'tagcn'}, default='stgcn'*  
  Specifies the name of method to be trained and evaluated. For each method, a different model is trained. 
- **stbln_symmetric**: *bool, default=False*  
  Specifies if the random graph in stbln method is symmetric or not. This parameter is used if method_name is 'stbln'. 
- **num_frames**: *int, default=300*  
  Specifies the number of frames in each skeleton sequence. This parameter is used if the method_name is 'tagcn'. 
- **num_subframes**: *int, default=100*  
  Specifies the number of sub-frames that are going to be selected by the tagcn model. This parameter is used if the method_name is 'tagcn'.   


#### `STGCNLearner.fit`
```python
STGCNLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose,
                 momentum, nesterov, weight_decay, train_data_filename,
                 train_labels_filename, val_data_filename,
                 val_labels_filename, skeleton_data_type)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Parameters:
  - **dataset**: *object*  
    Object that holds the training dataset.
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
  - **val_dataset**: *object*
    Object that holds the validation dataset. 
  - **logging_path**: *str, default=''*  
    Path to save TensorBoard log files and the training log files.
    If set to None or '', TensorBoard logging is disabled and no log file is created. 
  - **silent**: *bool, default=False*  
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True***  
    If set to True, enables the maximum verbosity.
  - **momentum**: *float, default=0.9*  
    Specifies the momentum value for optimizer. 
  - **nesterov**: *bool, default=True***  
    If set to true, the optimizer uses nesterov.  
  - **weight_decay**: *float, default=0.0001***  
    Specifies the weight_decay value of the optimizer. 
  - **train_data_filename**: *str, default='train_joints.npy'*  
    Filename that contains the training data. 
    This file should be contained in the dataset path provided.
    Note that this is a file name, not a path.
  - **train_labels_filename**: *str, default='train_labels.pkl'*  
    Filename of the labels .pkl file. 
    This file should be contained in the dataset path provided.
  - **val_data_filename**: *str, default='val_joints.npy'*  
    Filename that contains the validation data.
    This file should be contained in the dataset path provided.
    Note that this is a filename, not a path.
  - **val_labels_filename**: *str, default='val_labels.pkl'*  
    Filename of the validation labels .pkl file.
    This file should be contained in the dataset path provided.
  - **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
    The data stream that should be used for training and evaluation. 
    
#### `STGCNLearner.eval`
```python
STGCNLearner.eval(self, val_dataset, epoch, silent, verbose,
                  val_data_filename, val_labels_filename, skeleton_data_type,
                  save_score, wrong_file, result_file, show_topk)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
- **val_dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **epoch**: *int, default=0*  
  The training epoch in which the model is evaluated. 
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.
- **val_data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
  
- **save_score**: *bool, default=False*  
  The data stream that should be used for training and evaluation. 
- **wrong_file**: *str, default=None*  
  The data stream that should be used for training and evaluation. 
- **result_file**: *str, default=None*  
  The data stream that should be used for training and evaluation. 
- **show_topk**: *list, default=[1, 5]*  
  The data stream that should be used for training and evaluation. 
  
  
#### `STGCNLearner.infer`
```python
STGCNLearner.infer(self, point_clouds)
```

This method is used to perform pose estimation on an image.
Returns a list of `engine.target.BoundingBox3DList` objects if the list of `engine.data.PointCloud` is given or a single `engine.target.BoundingBox3DList` if a single `engine.data.PointCloud` is given.

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
  from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from perception.object_detection_3d.datasets.kitti import KittiDataset

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "perception", "object_detection_3d",
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
  from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from perception.object_detection_3d.datasets.kitti import LabeledPointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "perception", "object_detection_3d",
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
  from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from engine.datasets import PointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "perception", "object_detection_3d",
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
  from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
  )
  from engine.datasets import PointCloudsDatasetIterator

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "tanet_car"
  config = os.path.join(
    ".", "src", "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "second_detector", "configs", "tanet",
    "car", "test_short.proto")
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, "test_fit_" + name)

  subsets_path = os.path.join(
    ".", "src", "perception", "object_detection_3d",
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



## References

<a id="1">[1]</a> 
[Yan, S., Xiong, Y., & Lin, D. (2018, April). Spatial temporal graph convolutional networks for skeleton-based action 
recognition. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).](
https://arxiv.org/abs/1609.02907)

<a id="2">[2]</a> 
[Heidari, N., & Iosifidis, A. (2020). Temporal Attention-Augmented Graph Convolutional Network for Efficient Skeleton-
Based Human Action Recognition. arXiv preprint arXiv: 2010.12221.](https://arxiv.org/abs/2010.12221)

<a id="3">[3]</a> 
[Heidari, N., & Iosifidis, A. (2020). On the spatial attention in Spatio-Temporal Graph Convolutional Networks for 
skeleton-based human action recognition. arXiv preprint arXiv: 2011.03833.](https://arxiv.org/abs/2011.03833)

<a id="4">[4]</a> 
[Heidari, N., & Iosifidis, A. (2020). Progressive Spatio-Temporal Graph Convolutional Network for Skeleton-Based Human 
Action Recognition. arXiv preprint arXiv:2011.05668.](https://arxiv.org/pdf/2011.05668.pdf)

<a id="5">[5]</a> 
[Shahroudy, A., Liu, J., Ng, T. T., & Wang, G. (2016). Ntu rgb+ d: A large scale dataset for 3d human activity analysis.
 In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1010-1019).](
 https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html)

<a id="6">[6]</a>
[Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). 
The kinetics human action video dataset. arXiv preprint arXiv:1705.06950.](https://arxiv.org/pdf/1705.06950.pdf) 

<a id="7">[7]</a>
[Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2d pose estimation using part affinity 
fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).](
https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html)