## lightweight_open_pose module

The *lightweight_open_pose* module contains the *LightweightOpenPoseLearner* class, which inherits from the abstract class *Learner*.

### Class LightweightOpenPoseLearner
Bases: `engine.learners.Learner`

The *LightweightOpenPoseLearner* class is a wrapper of the Open Pose[[1]](#open-pose-1) implementation found on [Lightweight Open Pose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) [[2]](#open-pose-2).
It can be used to perform human pose estimation on images (inference) and train new pose estimation models.

The [LightweightOpenPoseLearner](/src/opendr/perception/lightweight_open_pose/lightweight_open_pose_learner.py) class has the
following public methods:

#### `LightweightOpenPoseLearner` constructor
```python
LightweightOpenPoseLearner(self, lr, epochs, batch_size, device, backbone, lr_schedule, temp_path, checkpoint_after_iter, checkpoint_load_iter, val_after, log_after, mobilenet_use_stride, mobilenetv2_width, shufflenet_groups, num_refinement_stages, batches_per_iter, experiment_name, num_workers, weights_only, output_name, multiscale, scales, visualize, base_height, img_mean, img_scale, pad_value, half_precision)
```

Constructor parameters:

- **lr**: *float, default=4e-5*\
  Specifies the initial learning rate to be used during training.
- **epochs**: *int, default=280*\
  Specifies the number of epochs the training should run for.
- **batch_size**: *int, default=80*\
  Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **backbone**: *{'mobilenet, 'mobilenetv2', 'shufflenet'}, default='mobilenet'*\
    Specifies the backbone architecture.
- **lr_schedule**: *str, default=' '*\
  Specifies the learning rate scheduler. Please provide a function that expects to receive as a sole argument the used optimizer.
- **temp_path**: *str, default='temp'*\
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files. 
  Moreover, the JSON file that contains the evaluation detections is saved here.
- **checkpoint_after_iter**: *int, default=5000*\
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **val_after**: *int, default=5000*\
  Specifies per how many training iterations a validation should be run.
- **log_after**: *int, default=100*\
  Specifies per how many training iterations the log files will be updated.
- **mobilenet_use_stride**: *bool, default=True*\
  Whether to add a stride value in the mobilenet model, which reduces accuracy but increases inference speed.
- **mobilenetv2_width**: *[0.0 - 1.0], default=1.0*\
  If the mobilenetv2 backbone is used, this parameter specifies its size.
- **shufflenet_groups**: *int, default=3*\
  If the shufflenet backbone is used, it specifies the number of groups to be used in grouped 1x1 convolutions in each ShuffleUnit.
- **num_refinement_stages**: *int, default=2*\
  Specifies the number of pose estimation refinement stages are added on the model's head, including the initial stage.
- **batches_per_iter**: *int, default=1*\
  Specifies per how many batches a backward optimizer step is performed.
- **experiment_name**: *str, default='default'*\
  String name to attach to checkpoints.
- **num_workers**: *int, default=8*\
  Specifies the number of workers to be used by the data loader.
- **weights_only**: *bool, default=True*\
  If True, only the model weights will be loaded; it won't load optimizer, scheduler, num_iter, current_epoch information.
- **output_name**: *str, default='detections.json'*\
  The name of the json file where the evaluation detections are stored, inside the temp_path.
- **multiscale**: *bool, default=False*\
  Specifies whether evaluation will run in the predefined multiple scales setup or not. 
  It overwrites self.scales to [0.5, 1.0, 1.5, 2.0].
- **scales**: *list, default=None*\
  A list of integer scales that define the multiscale evaluation setup. 
  Used to manually set the scales instead of going for the predefined multiscale setup.
- **visualize**: *bool, default=False*\
  Specifies whether the images along with the poses will be shown, one by one during evaluation.
- **base_height**: *int, default=256*\
  Specifies the height, based on which the images will be resized before performing the forward pass.
- **img_mean**: *list, default=(128, 128, 128)]*\
  Specifies the mean based on which the images are normalized.
- **img_scale**: *float, default=1/256*\
  Specifies the scale based on which the images are normalized.
- **pad_value**: *list, default=(0, 0, 0)*\
  Specifies the pad value based on which the images' width is padded.
- **half_precision**: *bool, default=False*\
  Enables inference using half (fp16) precision instead of single (fp32) precision. Valid only for GPU-based inference.

#### `LightweightOpenPoseLearner.fit`
```python
LightweightOpenPoseLearner.fit(self, dataset, val_dataset, logging_path, logging_flush_secs, silent, verbose, epochs, use_val_subset, val_subset_size, images_folder_name, annotations_filename)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.

Parameters:

  - **dataset**: *object*\
    Object that holds the training dataset.
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
  - **val_dataset**: *object, default=None*\
    Object that holds the validation dataset.
  - **logging_path**: *str, default=''*\
    Path to save TensorBoard log files.
    If set to None or '', TensorBoard logging is disabled.
  - **logging_flush_secs**: *int, default=30*\
    How often, in seconds, to flush the TensorBoard data to disk.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True***\
    If set to True, enables the maximum verbosity.
  - **epochs**: *int, default=None*\
    Overrides epochs attribute set in constructor.
  - **use_val_subset**: *bool, default=True***\
    If set to True, a subset of the validation dataset is created and used in evaluation.
  - **val_subset_size**: *int, default=250***\
    Controls the size of the validation subset.
  - **images_folder_name**: *str, default='train2017'*\
    Folder name that contains the dataset images.
    This folder should be contained in the dataset path provided.
    Note that this is a folder name, not a path.
  - **annotations_filename**: *str, default='person_keypoints_train2017.json'*\
    Filename of the annotations JSON file.
    This file should be contained in the dataset path provided.
  - **val_images_folder_name**: *str, default='val2017'*\
    Folder name that contains the validation images.
    This folder should be contained in the dataset path provided.
    Note that this is a folder name, not a path.
  - **val_annotations_filename**: *str, default='person_keypoints_val2017.json'*\
    Filename of the validation annotations JSON file.
    This file should be contained in the dataset path provided.

#### `LightweightOpenPoseLearner.eval`
```python
LightweightOpenPoseLearner.eval(self, dataset, silent, verbose, use_subset, subset_size, images_folder_name, annotations_filename)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing statistics regarding evaluation.

Parameters:

- **dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*\
  If set to True, enables the maximum verbosity.
- **use_subset**: *bool, default=True*\
  If set to True, a subset of the validation dataset is created and used in evaluation.
- **subset_size**: *int, default=250*\
  Controls the size of the validation subset.
- **images_folder_name**: *str, default='val2017'*\
  Folder name that contains the dataset images.
  This folder should be contained in the dataset path provided.
  Note that this is a folder name, not a path.
- **annotations_filename**: *str, default='person_keypoints_val2017.json'*\
  Filename of the annotations JSON file.
  This file should be contained in the dataset path provided.

#### `LightweightOpenPoseLearner.infer`
```python
LightweightOpenPoseLearner.infer(self, img, upsample_ratio, track, smooth)
```

This method is used to perform pose estimation on an image.
Returns a list of `engine.target.Pose` objects, where each holds a pose, or returns an empty list if no detections were made.

Parameters:

- **img**: *object***\
  Object of type engine.data.Image.
- **upsample_ratio**: *int, default=4*\
  Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing.
- **track**: *bool, default=True*\
  If True, infer propagates poses ids from previous frame results to track poses.
- **smooth**: *bool, default=True*\
  If True, smoothing is performed on pose keypoints between frames.

#### `LightweightOpenPoseLearner.save`
```python
LightweightOpenPoseLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already
exist. Inside this folder, the model is saved as "name.pth" and the metadata file as "name.json". If the directory
already exists, the "name.pth" and "name.json" files are overwritten.

If [`self.optimize`](#LightweightOpenPoseLearner.optimize) was run previously, it saves the optimized ONNX model in
a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously
during conversion.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `LightweightOpenPoseLearner.load`
```python
LightweightOpenPoseLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `LightweightOpenPoseLearner.optimize`
```python
LightweightOpenPoseLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export. Constant-folding optimization will replace some of the ops that have all constant inputs with pre-computed constant nodes.

#### `LightweightOpenPoseLearner.download`
```python
LightweightOpenPoseLearner.download(self, path, mode, verbose, url)
```

Download utility for various Lightweight Open Pose components. Downloads files depending on mode and
saves them in the path provided. It supports the following modes:
1. "pretrained": the default mobilenet pretrained model
2. "weights": mobilenet, mobilenetv2 and shufflenet weights needed for training
3. "test_data": a test dataset with a single COCO image and its annotation

Parameters:

- **path**: *str, default=None*\
  Local path to save the files, defaults to self.temp_path if None.
- **mode**: *str, default="pretrained"*\
  What file to download, can be one of "pretrained", "weights", "test_data"
- **verbose**: *bool, default=False*\
  Whether to print messages in the console.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### Examples

* **Training example using an `ExternalDataset`**.
  To train properly, the backbone weights are downloaded automatically in the `temp_path`. Default backbone is
  'mobilenet'.
  The training and evaluation dataset should be present in the path provided, along with the JSON annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from opendr.perception.pose_estimation import LightweightOpenPoseLearner
  from opendr.engine.datasets import ExternalDataset

  pose_estimator = LightweightOpenPoseLearner(temp_path='./parent_dir', batch_size=8, device="cuda",
                                              num_refinement_stages=3)

  training_dataset = ExternalDataset(path="./data", dataset_type="COCO")
  validation_dataset = ExternalDataset(path="./data", dataset_type="COCO")
  pose_estimator.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path="./logs")
  pose_estimator.save('./saved_models/trained_model')
  ```

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  import cv2
  from opendr.perception.pose_estimation import LightweightOpenPoseLearner
  from opendr.perception.pose_estimation import draw, get_bbox
  from opendr.engine.data import Image

  pose_estimator = LightweightOpenPoseLearner(device="cuda", temp_path='./parent_dir')
  pose_estimator.download()  # Download the default pretrained mobilenet model in the temp_path
  pose_estimator.load("./parent_dir/mobilenet_openpose")
  pose_estimator.download(mode="test_data")  # Download a test data taken from COCO2017

  img = Image.open('./parent_dir/dataset/image/000000000785.jpg')
  orig_img = img.opencv()  # Keep original image
  current_poses = pose_estimator.infer(img)
  img_opencv = img.opencv()
  for pose in current_poses:
      draw(img_opencv, pose)
  img_opencv = cv2.addWeighted(orig_img, 0.6, img_opencv, 0.4, 0)
  cv2.imshow('Result', img_opencv)
  cv2.waitKey(0)
  ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from opendr.perception.pose_estimation import LightweightOpenPoseLearner

  pose_estimator = LightweightOpenPoseLearner(temp_path='./parent_dir')

  pose_estimator.download()  # Download the default pretrained mobilenet model in the temp_path
  pose_estimator.load("./parent_dir/mobilenet_openpose")
  pose_estimator.optimize(do_constant_folding=True)
  pose_estimator.save('./parent_dir/optimized_model')
  ```

#### Performance Evaluation

The performance evaluation results of the *LightweightOpenPoseLearner* are reported in the Table below:

| Method            | CPU i7-9700K (FPS) | RTX 2070 (FPS) | Jetson TX2 (FPS) | Xavier NX (FPS) | Xavier AGX (FPS) |
|-------------------|-------|-------|-----|-----|-------|
| OpenDR - Baseline | 13.6  | 50.1  | 5.6 | 7.4 | 11.2  |
| OpenDR - Half     | 13.5  | 47.5  | 5.4 | 9.5 | 12.9  |
| OpenDR - Stride   | 31.0  | 72.1  | 12.2| 13.8| 15.5  |
| OpenDR - Stages   | 19.3  | 59.8  | 7.2 | 10.2| 15.7  |
| OpenDR - H+S      | 30.9  | 68.4  | 12.2| 12.9| 18.4  |
| OpenDR - Full     | 47.4  | 98.2  | 17.1| 18.8| 25.9  |

We have evaluated the effect of using different inference settings, namely:
- *OpenDR - Baseline*, which refers to directly using the Lightweight OpenPose method adapted to OpenDR with no additional optimizations,
- *OpenDR - Half*, which refers to enabling inference in half (FP) precision,
- *OpenDR - Stride*, which refers to increasing stride by two in the input layer of the model,
- *OpenDR - Stages*, which refers to removing the refinement stages,
- *OpenDR - H+S*, which uses both half precision and increased stride, and
- *OpenDR - Full*, which refers to combining all three available optimization.

Apart from the inference speed, which is reported in FPS, we also report the memory usage, as well as energy consumption on a reference platform in the Table below:

| Method  | Memory (MB) | Energy (Joules)  - Total per inference  |
|-------------------|---------|-------|
| OpenDR - Baseline | 1187.75 | 1.65  | 
| OpenDR - Half     | 1085.75 | 1.17  |
| OpenDR - Stride   | 1037.5  | 1.40  |
| OpenDR - Stages   | 1119.75 | 1.15  |
| OpenDR - H+S      | 1013.75 | 0.70  |
| OpenDR - Full     | 999.75  | 0.52  |

NVIDIA Jetson AGX was used as the reference platform for measuring energy requirements for these experiments. 
We calculated the average metrics of 100 runs, while an image with resolution of 640×425 pixels was used as input to the models. 

The average precision and average recall on the COCO evaluation split is also reported in the Table below:

| Method |  Average Precision (IoU=0.50)  | Average Recall (IoU=0.50) |
|-------------------|-------|-------|
| OpenDR - Baseline | 0.557 | 0.598 |
| OpenDR - Half     | 0.558 | 0.594 |
| OpenDR - Stride   | 0.239 | 0.283 |
| OpenDR - Stages   | 0.481 | 0.527 |
| OpenDR - H+S      | 0.240 | 0.281 |
| OpenDR - Full     | 0.203 | 0.245 |

For measuring the precision and recall we used the standard approach proposed for COCO, using an Intersection of Union (IoU) metric at 0.5. 

The platform compatibility evaluation is also reported below:

| Platform  | Compatibility Evaluation |
| ----------------------------------------------|-------|
| x86 - Ubuntu 20.04 (bare installation - CPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (bare installation - GPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (pip installation)         | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (CPU docker)               | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (GPU docker)               | :heavy_check_mark:   |
| NVIDIA Jetson TX2                             | :heavy_check_mark:   |
| NVIDIA Jetson Xavier AGX                      | :heavy_check_mark:   |
| NVIDIA Jetson Xavier NX                       | :heavy_check_mark:   |


#### Notes

For the metrics of the algorithm the COCO dataset evaluation scores are used as explained [here](
https://cocodataset.org/#keypoints-eval).

Keypoints and how poses are constructed is according to the original method described [here](
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/TRAIN-ON-CUSTOM-DATASET.md).

Pose keypoints ids are matched as:

| Keypoint ID 	| Keypoint name  	| Keypoint abbrev. 	|
|-------------	|----------------	|------------------	|
| 0           	| nose           	| nose             	|
| 1           	| neck           	| neck             	|
| 2           	| right shoulder 	| r_sho            	|
| 3           	| right elbow    	| r_elb            	|
| 4           	| right wrist    	| r_wri            	|
| 5           	| left shoulder  	| l_sho            	|
| 6           	| left elbow     	| l_elb            	|
| 7           	| left wrist     	| l_wri            	|
| 8           	| right hip      	| r_hip            	|
| 9           	| right knee     	| r_knee           	|
| 10          	| right ankle    	| r_ank            	|
| 11          	| left hip       	| l_hip            	|
| 12          	| left knee      	| l_knee           	|
| 13          	| left ankle     	| l_ank            	|
| 14          	| right eye      	| r_eye            	|
| 15          	| left eye       	| l_eye            	|
| 16          	| right ear      	| r_ear            	|
| 17          	| left ear       	| l_ear            	|


#### References
<a name="open-pose-1" href="https://arxiv.org/abs/1812.08008">[1]</a> OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,
[arXiv](https://arxiv.org/abs/1812.08008).
<a name="open-pose-2" href="https://arxiv.org/abs/1811.12004">[2]</a> Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,
[arXiv](https://arxiv.org/abs/1811.12004).
