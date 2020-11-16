## lightweight_open_pose module

The *lightweight_open_pose* module contains the *LightweightOpenPoseLearner* class, which inherits from the abstract 
class *Learner*.

### Class LightweightOpenPoseLearner
Bases: `engine.learners.Learner`

The *LightweightOpenPoseLearner* class is a wrapper of the Open Pose[[1]](#1) implementation found on 
[Lightweight Open Pose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) [[2]](#2). It can 
be used to perform human pose estimation on images (inference) and train new pose estimation models.

The [LightweightOpenPoseLearner](#src.perception.lightweight_open_pose.lightweight_open_pose_learner) class uses the 
the following arguments in its constructor:

|Parameters:| | 
|:---|:-------------|
| |**lr: *float, default=4e-5*** <br /> &nbsp; &nbsp; &nbsp;Specifies the initial learning rate to be used during training.|
| |**epochs: *int, default=280*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of epochs the training should run for.| 
| |**batch_size: *int, default=80*** <br /> &nbsp; &nbsp; &nbsp;Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.|
| |**device: *{'cpu', 'cuda'}, default='cuda'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the device to be used. |
| |**backbone: *{'mobilenet, 'mobilenetv2', 'shufflenet'}, default='mobilenet'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the backbone architecture.|
| |**lr_schedule: *str, default=' '*** <br /> &nbsp; &nbsp; &nbsp;Specifies the learning rate scheduler.|
| |**temp_path: *str, default='temp'*** <br /> &nbsp; &nbsp; &nbsp;Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files. Moreover the json file that contains the evaluation detections is saved here.|
| |**checkpoint_after_iter: *int, default=5000*** <br /> &nbsp; &nbsp; &nbsp;Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.|  
| |**checkpoint_load_iter: *int, default=0*** <br /> &nbsp; &nbsp; &nbsp;Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.|    
| |**val_after: *int, default=5000*** <br /> &nbsp; &nbsp; &nbsp;Specifies per how many training iterations a validation should be run.|          
| |**log_after: *int, default=100*** <br /> &nbsp; &nbsp; &nbsp;Specifies per how many training iterations the log files will be updated.|          
| |**mobilenetv2_width: *[0.0 - 1.0], default=1.0*** <br /> &nbsp; &nbsp; &nbsp;If the mobilenetv2 backbone is used, this parameter specified its size.|          
| |**shufflenet_groups: *int, default=3*** <br /> &nbsp; &nbsp; &nbsp;If the shufflenet backbone is used, it specifieds the number of groups to be used in grouped 1x1 convolutions in each ShuffleUnit.|      
| |**num_refinement_states: *int, default=3*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of pose estimation refinement stages are added on the model's head, including the initial stage.|
| |**batches_per_iter: *int, default=1*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of batches from which losses will be calculated before running a single optimization step.|
| |**experiment_name: *str, default='default'*** <br /> &nbsp; &nbsp; &nbsp;String name to attach to checkpoints.|
| |**num_workers: *int, default=8*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of workers to be used by the data loader.|
| |**weights_only: *bool, default=True*** <br /> &nbsp; &nbsp; &nbsp;If True, only the model weights will be loaded; it won't load optimizer, scheduler, num_iter, current_epoch information.|
| |**output_name: *str, default='detections.json'*** <br /> &nbsp; &nbsp; &nbsp;The name of the json files where the evaluation detections are stored, inside the temp_path.|
| |**multiscale: *bool, default=False*** <br /> &nbsp; &nbsp; &nbsp;Specifies whether evaluation will run in the predefined multiple scales setup or not. It overwrites self.scales to [0.5, 1.0, 1.5, 2.0].|
| |**scales: *list, default=[1]*** <br /> &nbsp; &nbsp; &nbsp;A list of integer scales that define the multiscale evaluation setup. Used to manually set the scales instead of going for the predefined multiscale setup.|
| |**visualize: *bool, default=False*** <br /> &nbsp; &nbsp; &nbsp;Specifies whether the images along with the poses will be shown, one by one during evaluation.|
| |**base_height: *int, default=256*** <br /> &nbsp; &nbsp; &nbsp;Specifies the height, based on which the images will be resized before performing the forward pass.|
| |**stride: *int, default=8*** <br /> &nbsp; &nbsp; &nbsp;Specifies the stride based on which padding will be performed.|
| |**pad_value: *list, default=[0, 0, 0]*** <br /> &nbsp; &nbsp; &nbsp;Specifies the pad value based on which the images' width is padded.|
| |**img_mean: *list, default=[128, 128, 128]*** <br /> &nbsp; &nbsp; &nbsp;Specifies the mean based on which the images are normalized.|
| |**img_scale: *float, default=1/256*** <br /> &nbsp; &nbsp; &nbsp;Specifies the scale based on which the images are normalized.|


   

The [LightweightOpenPoseLearner](#src.perception.lightweight_open_pose.lightweight_open_pose_learner.py) class has the 
following public methods:

---

|fit(dataset, val_dataset, logging_path, logging_flush_secs, silent, verbose, epochs, use_val_subset, val_subset_size, images_folder_name, annotations_filename)|
|:---|
This method is used for training the algorithm on a train dataset and validating on a val dataset.

| | | 
|:---|:-------------|
|Parameters:  | **dataset: *object*** <br /> Object that holds the training dataset. Can be of type *ExternalDataset* or a custom dataset inheriting from *DatasetIterator*.|
| | **val_dataset: *object*** <br /> Object that holds the validation dataset.|
| | **logging_path: *str, default=''*** <br /> Path to save tensorboard log files. If set to None or '', tensorboard logging is disabled.|
| | **logging_flush_secs: *int, default=30*** <br />  How often, in seconds, to flush the tensorboard data to disk.|
| | **silent: *bool, default=False*** <br /> If set to True, disables all printing of training progress reports and other information to STDOUT.|
| | **verbose: *bool, default=True*** <br /> If set to True, enables the maximum verbosity.|
| | **epochs: *int, default=None*** <br /> Overrides epochs attribute set in constructor.|
| | **use_val_subset: *bool, default=True*** <br /> If set to True, a subset of the validation dataset is created and used in evaluation.|
| | **val_subset_size: *int, default=250*** <br /> Controls the size of the validation subset.|
| | **images_folder_name: *str, default='train2017'*** <br /> Folder name that contains the dataset images. This folder should be contained in the dataset path provided. Note that this is a folder name, not a path.|
| | **annotations_filename: *str, default='person_keypoints_train2017.json'*** <br /> Filename of the annotations json file. This file should be contained in the dataset path provided.|
|**Returns**: | ***dict***<br />Returns stats regarding the last evaluation ran.| 

---

|eval(dataset, silent, verbose, use_subset, subset_size, images_folder_name, annotations_filename)|
|:---|
This method is used to evaluate a trained model on an evaluation dataset.

|| | 
|:---|:-------------|
| Parameters: | **dataset: *object*** <br /> Object that holds the evaluation dataset. Can be of type *ExternalDataset* or a custom dataset inheriting from *DatasetIterator*.| 
| | **silent: *bool, default=False*** <br /> If set to True, disables all printing of evalutaion progress reports and other information to STDOUT.|
| | **verbose: *bool, default=True*** <br /> If set to True, enables the maximum verbosity.|
| | **val_subset: *bool, default=True*** <br /> If set to True, a subset of the validation dataset is created and used in evaluation.|
| | **subset_size: *int, default=250*** <br /> Controls the size of the validation subset.|
| | **images_folder_name: *str, default='val2017'*** <br /> Folder name that contains the dataset images. This folder should be contained in the dataset path provided. Note that this is a folder name, not a path.|
| | **annotations_filename: *str, default='person_keypoints_val2017.json'*** <br /> Filename of the annotations json file. This file should be contained in the dataset path provided.|
|**Returns**: | ***dict***<br />Returns stats regarding evaluation |

---

|infer(img, upsample_ratio, track, smooth)|
|:---|
This method is used to perform pose estimation on an image.

| | | 
|:---|:-------------|
| Parameters: | **img: *object*** <br /> Object of type engine.data.Image. |
| | **upsample_ratio: *int, default=4*** <br /> Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing.|
| | **track: *bool, default=True*** <br /> If True, infer propagates poses ids from previous frame results to track poses.|
| | **smooth: *bool, default=True*** <br /> If True, smoothing is performed on pose keypoints between frames.|
|**Returns**: |***list*** <br /> Returns a list of engine.target.Pose objects, where each holds a pose, or returns an empty list if no detections were made.|  

---

|save(path)|
|:---|
This method is used to save a trained model. Saves the current model's *state_dict*.
If *self.optimize* was ran earlier, this method saves the ONNX model in the path provided.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> Path to save the model.|

---

|load(path)|
|:---|
This method is used to load a trained model. Loads the *state_dict* saved with the *save* method.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> Path of the model to be loaded.|

---

|load_from_onnx(path)|
|:---|
This method is used to load an optimized ONNX model previously saved.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> Path of the onnx model to be loaded.

---

|optimize(do_constant_folding)|
|:---|
This method is used to optimize a trained model to ONNX format which can be then used for inference.

|| | 
|:---|:-------------|
| Parameters: | **do_constant_folding: *bool, default=False*** <br />  ONNX format optimization. If True, the constant-folding optimization is applied to the model during export. Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.|

---

**Examples**

---

*Training example using an ExternalDataset. To train properly, the backbone weights need to be present in the defined 
temp_path. Default backbone is 'mobilenet', whose weights can be found in this [Google Drive](
https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view). The training and evaluation dataset should 
be present in the path provided, along with the .json annotation files. The default COCO 2017 training data can be 
found [here](https://cocodataset.org/#download) (train, val, annotations).*
```python
from OpenDR.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from OpenDR.engine.datasets import ExternalDataset

pose_estimator = LightweightOpenPoseLearner(temp_path='./parent_dir', batch_size=8, device="cuda", 
                                            num_refinement_stages=3)

training_dataset = ExternalDataset(path="./data", dataset_type="COCO")
validation_dataset = ExternalDataset(path="./data", dataset_type="COCO")
pose_estimator.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path="./logs")
pose_estimator.save('./saved_models/trained_model.pth')
```

---

*Inference and result drawing example on a test .jpg image using OpenCV.*
```python
import cv2
from OpenDR.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from OpenDR.perception.pose_estimation.lightweight_open_pose.utilities import draw, get_bbox

pose_estimator = LightweightOpenPoseLearner(device="cuda")
pose_estimator.load("./trained_models/mobilenetModel.pth")
img = cv2.imread('./test.jpg')
orig_img = img.copy()  # Keep original image
current_poses = pose_estimator.infer(img)
for pose in current_poses:
    draw(img, pose)
img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
cv2.imshow('Result', img)
cv2.waitKey(0)
```

---

*Optimization example for a previously trained model. Inference can be run with the trained model after running 
self.optimize.*
```python
from OpenDR.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner

pose_estimator = LightweightOpenPoseLearner(temp_path='./parent_dir')
pose_estimator.load("./trained_model.pth")
pose_estimator.optimize(do_constant_folding=True)
pose_estimator.save('./saved_models/optimized_model.onnx')
```

---

**Notes**

---

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


- <a id="1">[1]</a>: OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, 
[arXiv](https://arxiv.org/abs/1812.08008).
- <a id="2">[2]</a>: Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose, 
[arXiv](https://arxiv.org/abs/1811.12004).
