## high_resolution_pose_estimation module

The *high_resolution_pose_estimation* module contains the *HighResolutionPoseEstimationLearner* class, which inherits from the abstract class *Learner*.

### Class HighResolutionPoseEstimationLearner
Bases: `engine.learners.Learner`

The *HighResolutionLightweightOpenPose* class is an implementation for pose estimation in high resolution images. This method is creating a heatmap of a resized version of the input image. Using this heatmap, the input image is being cropped keeping the area of interest and then it isused for pose estimation. The pose estimation is based on the Lightweight OpenPose algorithm. 
In this method there are two important variables which are responsible for the increase in speed and in accuracy in high resolution images. These variables are the *first_pass_height* and the *second height* that the image is resized in this procedure.

The [HighResolutionPoseEstiamtionLearner](/src/opendr/perception/pose_estimation/hr_pose_estimation/HighResolutionLearner.py) class has the
following public methods:

#### `HighResolutionPoseEstimationLearner` constructor
```python
HighResolutionPoseEstimationLearner(self,device, backbone, temp_path, mobilenet_use_stride, mobilenetv2_width, shufflenet_groups, num_refinement_stages, batches_per_iter, base_height, first_pass_height, second_pass_height, img_resol, experiment_name, num_workers, weights_only, output_name, multiscale, scales, visualize,  img_mean, img_scale, pad_value)
```

Constructor parameters:

- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **backbone**: *{'mobilenet, 'mobilenetv2', 'shufflenet'}, default='mobilenet'*\
    Specifies the backbone architecture.
- **temp_path**: *str, default='temp'*\
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files. Moreover the JSON file that contains the evaluation detections is saved here.
- **mobilenet_use_stride**: *bool, default=True*\
  Whether to add an additional stride value in the mobilenet model, which reduces accuracy but increases inference speed.
- **mobilenetv2_width**: *[0.0 - 1.0], default=1.0*\
  If the mobilenetv2 backbone is used, this parameter specified its size.
- **shufflenet_groups**: *int, default=3*\
  If the shufflenet backbone is used, it specifies the number of groups to be used in grouped 1x1 convolutions in each ShuffleUnit.
- **num_refinement_stages**: *int, default=2*\
  Specifies the number of pose estimation refinement stages are added on the model's head, including the initial stage.
- **batches_per_iter**: *int, default=1*\
  Specifies per how many batches a backward optimizer step is performed.
- **base_height**: *int, default=256*\
  Specifies the height, based on which the images will be resized before performing the forward pass when using the Lightweight OpenPose.
- **first_pass_height**: *int, default=360*\
  Specifies the height that the input image will be resized during the heatmap generation procedure.
- **second_pass_heght**: *int, default=540*\
  Specifies the height of the image on the second inference for pose estimation procedure.
- **experiment_name**: *str, default='default'*\
  String name to attach to checkpoints.
- **num_workers**: *int, default=8*\
  Specifies the number of workers to be used by the data loader.
- **weights_only**: *bool, default=True*\
  If True, only the model weights will be loaded; it won't load optimizer, scheduler, num_iter, current_epoch information.
- **output_name**: *str, default='detections.json'*\
  The name of the json files where the evaluation detections are stored, inside the temp_path.
- **multiscale**: *bool, default=False*\
  Specifies whether evaluation will run in the predefined multiple scales setup or not. It overwrites self.scales to [0.5, 1.0, 1.5, 2.0].
- **scales**: *list, default=None*\
  A list of integer scales that define the multiscale evaluation setup. Used to manually set the scales instead of going for the predefined multiscale setup.
- **visualize**: *bool, default=False*\
  Specifies whether the images along with the poses will be shown, one by one during evaluation.

- **img_mean**: *list, default=(128, 128, 128)]*\
  Specifies the mean based on which the images are normalized.
- **img_scale**: *float, default=1/256*\
  Specifies the scale based on which the images are normalized.
- **pad_value**: *list, default=(0, 0, 0)*\
  Specifies the pad value based on which the images' width is padded.
- **half_precision**: *bool, default=False*\
  Enables inference using half (fp16) precision instead of single (fp32) precision. Valid only for GPU-based inference.



#### `HighResolutionPoseEstimationLearner.eval`
```python
HighResolutionPoseEstimationLearner.eval(self, dataset, silent, verbose, use_subset, subset_size, images_folder_name, annotations_filename)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:

- **dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*\
  If set to True, enables the maximum verbosity.
- **val_subset**: *bool, default=True*\
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

#### `HighResolutionPoseEstimation.infer`
```python
HighResolutionPoseEstimation.infer(img, upsample_ratio, track, smooth)
```

This method is used to perform pose estimation on an image.
Returns a list of `engine.target.Pose` objects, where each holds a pose, or returns an empty list if no detection were made.

Parameters:

- **img**: *object***\
  Object of type engine.data.Image.
- **upsample_ratio**: *int, default=4*\
  Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing.
- **track**: *bool, default=True*\
  If True, infer propagates poses ids from previous frame results to track poses.
- **smooth**: *bool, default=True*\
  If True, smoothing is performed on pose keypoints between frames.


#### `HighResolutionPoseEstimationLearner.first_pass`
```python
HighResolutionPoseEstimationLearner.first_pass(self, net,img)
```
This method is used for extracting from the input image a heatmap about human locations in the picture.

Parameters:
- **img**: *object***\
  Object of type engine.data.Image.
- **net**: *object*\
  The model that is used for creating the heatmap.

#### `HighResolutionPoseEstimationLearner.second_pass`
```python
HighResolutionPoseEstimationLearner.second_pass_infer(self, net, img, net_input_height_size, max_width, stride, upsample_ratio, device,
                          pad_value=(0, 0, 0),
                          img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256))
```
On this method it is carried out the second inference step which estimates the human poses on the image that is inserted. Following the steps of the proposed method this image should be the cropped part of the initial high resolution image that came out from taking into account the area of interest of heatmap generation.

Parameters:
- **net**: *object*\
  The model that is used for creating the heatmap.
- **img**: *object***\
  Object of type engine.data.Image.
- **net_input_height_size**: *int*\
    It is the height that is used for resizing the image on the pose estimation procedure.
- **max_width**: *int*\
   It is the max width that the cropped image should have in order to keep the height-width ratio below a certain value.
- **stride**: *int*\
   Is the stride value of mobilenet which reduces accuracy but increases inference speed.
- **upsample_ratio**: *int, default=4*\
  Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing.

- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **pad_value**: *list, default=(0, 0, 0)*\
  Specifies the pad value based on which the images' width is padded.
- **img_mean**: *list, default=(128, 128, 128)]*\
  Specifies the mean based on which the images are normalized.
- **img_scale**: *float, default=1/256*\
  Specifies the scale based on which the images are normalized.

#### `HighResolutionPoseEstimation.Pooling`
```python
HighResolutionPoseEstimation.Pooling(self, img, kernel)
```
Parameters:
- **img**: *object***\
  Object of type engine.data.Image.
- **kernel**: *int*\
  The size of kernel in Average Pooling procedure before heatmap generation in order to resize the input image.

#### `HighResolutionPoseEstimation.infer_light_odr()`
```python
HighResolutionPoseEstimation.infer_light_odr(self, img, upsample_ratio=4, track=True, smooth=True)
```
This method is an inference function of OpenDR LightWeight OpenPose pose estiamtion.

Parameters:

- **img**: *object***\
  Object of type engine.data.Image.
- **upsample_ratio**: *int, default=4*\
  Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing.
- **track**: *bool, default=True*\
  If True, infer propagates poses ids from previous frame results to track poses.
- **smooth**: *bool, default=True*\
  If True, smoothing is performed on pose keypoints between frames.

#### `HighResolutionPoseEstiamtion.save`
```python
HighResolutionPoseEstimation.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already
exist. Inside this folder, the model is saved as "name.pth" and the metadata file as "name.json". If the directory
already exists, the "name.pth" and "name.json" files are overwritten.



Parameters:

- **path**: *str*\
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `HighResolutionPoseEstimation.load`
```python
HighResolutionPoseEstimation.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.
- **verbose**: *bool, default=False*\
  If set to True, prints a message on success.

#### `HighResolutionPoseEstimation.download`
```python
HighResolutionPoseEstimation.download(self, path, mode, verbose, url)
```

Download utility for various Lightweight Open Pose components. Downloads files depending on mode and
saves them in the path provided. It supports downloading:
1. the default mobilenet pretrained model
2. mobilenet, mobilenetv2 and shufflenet weights needed for training
3. a test dataset with a single COCO image and its annotation

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

* **Inference and result drawing example on a test .jpg image using OpenCV.**
  ```python
  import cv2
  from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner
  from opendr.perception.pose_estimation import draw
  from opendr.engine.data import Image
  
  pose_estimator = HighResolutionPoseEstimationLearner(device='cuda', num_refinement_stages=2,
                                                           mobilenet_use_stride=False, half_precision=False,
                                                           first_pass_height=360,
                                                           second_pass_height=540,
                                                           img_resol=1080)
  pose_estimator.download()  # Download the default pretrained mobilenet model in the temp_path
  
  pose_estimator.load("./parent_dir/openpose_default")
  pose_estimator.download(mode="test_data")  # Download a test data taken from COCO2017
  
  img = Image.open('./parent_dir/dataset/image/000000000785_1080.jpg')
  orig_img = img.opencv()  # Keep original image
  current_poses = pose_estimator.infer(img)
  img_opencv = img.opencv()
  for pose in current_poses:
      draw(img_opencv, pose)
  img_opencv = cv2.addWeighted(orig_img, 0.6, img_opencv, 0.4, 0)
  cv2.imshow('Result', img_opencv)
  cv2.waitKey(0)
  ```


#### Performance Evaluation


In order to check the performance of the *HighResolutionPoseEstimationLearner* it had been tested on various platforms and with different optimizations that Lightweight OpenPose offers.
The experiments are conducted on a 1080p image.
#### Lightweight OpenPose With resizing on 256 pixels
|                    **Method**                    | **CPU i7-9700K (FPS)** | **RTX 2070 (FPS)** | **Jetson TX2 (FPS)** | **Xavier NX (FPS)** |
|:------------------------------------------------:|-----------------------|-------------------|----------------------|---------------------|
|                OpenDR - Baseline                 | 0.9                   | 46.3              | 4.6                  | 6.4                 |
|                  OpenDR - Full                   | 2.9                   | 83.1              | 11.2                 | 13.5                |


#### Lightweght OpenPoseWithout resizing
| Method            | CPU i7-9700K (FPS) | RTX 2070  (FPS) | Jetson TX2 (FPS) | Xavier NX (FPS) |
|-------------------|--------------------|-----------------|------------------|-----------------|
| OpenDR - Baseline | 0.05               | 2.6             | 0.3              | 0.5             |
| OpenDR - Full     | 0.2                | 10.8            | 1.4              | 3.1             |

#### High-Resolution Pose Estimation
| Method                 | CPU i7-9700K (FPS) | RTX 2070 (FPS) | Jetson TX2 (FPS) | Xavier NX (FPS) |
|------------------------|--------------------|----------------|------------------|-----------------|
| HRPoseEstim - Baseline | 2.3                | 13.6           | 1.4              | 1.8             |
| HRPoseEstim - Half     | 2.7                | 16.1           | 1.3              | 3.0             |
| HRPoseEstim - Stride   | 8.1                | 27.0           | 4                | 4.9             |
| HRPoseEstim - Stages   | 3.7                | 16.5           | 1.9              | 2.7             |
| HRPoseEstim - H+S      | 8.2                | 25.9           | 3.6              | 5.5             |
| HRPoseEstim - Full     | 10.9               | 31.7           | 4.8              | 6.9             |


As it is shown in the previous Table, OpenDr Lightweight OpenPose achieves higher FPS when it is resing the input image into 256 pixels. It is easier to process that image but as it is shown in the next tables the method falls apart when it comes to accuracy and there are no detections.

We have evaluated the effect of using different inference settings, namely:
- *HRPoseEstim - Baseline*, which refers to directly using the High Resolution Pose Estimation method,which is based in Lightweight OpenPose,
- *HRPoseEstim - Half*, which refers to enabling inference in half (FP) precision,
- *HRPoseEstim - Stride*, which refers to increasing stride by two in the input layer of the model,
- *HRPoseEstim - Stages*, which refers to removing the refinement stages,
- *HRPoseEstim - H+S*, which uses both half precision and increased stride, and
- *HRPoseEstim - Full*, which refers to combining all three available optimization.
was used as input to the models. 

The average precision and average recall on the COCO evaluation split is also reported in the Table below:
#### Lightweight OpenPose
| Method            | Average Precision (IoU=0.50) | Average Recall (IoU=0.50) | Evaluation time (sec) |
|-------------------|------------------------------|---------------------------|-----------------------|
| OpenDR - Baseline | 0.0                          | 0.0                       | 6558                  |


#### High Resolution Pose Estimation
| Method                 | Average Precision (IoU=0.50) | Average Recall (IoU=0.50) | Evaluation Time (sec) |
|------------------------|------------------------------|---------------------------|-----------------------|
| HRPoseEstim - Baseline | 0.615                        | 0.637                     | 288                   |
| HRPoseEstim - Half     | 0.604                        | 0.621                     | 269                   |
| HRPoseEstim - Stride   | 0.262                        | 0.274                     | 160                   |
| HRPoseEstim - Stages   | 0.539                        | 0.562                     | 238                   |
| HRPoseEstim - H+S      | 0.254                        | 0.267                     | 165                   |
| HRPoseEstim - Full     | 0.259                        | 0.272                     | 145                   |



The average precision and the average recall have been calculated on a 1080p version of COCO2017 validation dataset and the results are reported in the table below:

| Method | Average Precision (IoU=0.50) | Average Recall (IoU=0.50) |
|-------------------|------------------------------|---------------------------|
| HRPoseEstim - Baseline | 0.518                        | 0.536                     |
| HRPoseEstim - Half     | 0.509                        | 0.520                     |
| HRPoseEstim - Stride   | 0.143                        | 0.149                     |
| HRPoseEstim - Stages   | 0.474                        | 0.496                     |
| HRPoseEstim - H+S      | 0.134                        | 0.139                     |
| HRPoseEstim - Full     | 0.141                        | 0.150                     |

For measuring the precision and recall we used the standard approach proposed for COCO, using an Intersection of Union (IoU) metric at 0.5. 


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
