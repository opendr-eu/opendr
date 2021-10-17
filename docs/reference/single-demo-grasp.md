# mobile_manipulation module

The *single_demo_grasp* module contains the *SingleDemoGraspLearner* class, which inherits from the abstract class *Learner*.

### Class SingleDemoGraspLearner
Bases: `engine.learners.Learner`

The *SingleDemoGraspLearner* class is a wrapper for keypoint_rcnn detector from [Detectron2](https://github.com/facebookresearch/detectron2).
It can be used to perform object and keypoint detection which can be later utilized for localizing and correcting the orientation of an object and to find the correct grasping pose.

The [SingleDemoGraspLearner](#src.opendr.control.single_demo_grasp.keypoint_detector_2d.training.single_demo_grasp_learner.py) class has the following 
public methods:

#### `SingleDemoGraspLearner` constructor
```python
SingleDemoGraspLearner(self, object_name = '', dataset_dir = '', lr = 0.0008, batch_size = 2,
                    num_workers = 2, num_classes = 1, iters = 1000, threshold = 0.8, device = 'cuda')
```



Constructor parameters:
- **lr**: *float, default=8e-4*  
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=1000*  
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=2*  
  Specifies the batch size during training.
- **num_workers**: *int, default=2*  
  number of parallel data loading workers (Detectron2 dataloader parameters)
- **num_classes**: *int, default=1*
  Specifies the number of object classes. The always 1 for the default parameters in constructor yo achieve good results.
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device to be used.
- **threshold**: *float, default=0.8*  
  Specifies threshold to filter outputs based on the score.
- **object_name**: *str, default=''*  
  Specifies name of the object. it is used internally to save and load data properly.
- **dataset_dir**: *str, default=''*  
  Specifies the directory in which the model should look for training/val datasets


#### `SingleDemoGraspLearner.fit`
```python
SingleDemoGraspLearner.fit(self)
```

Starts training on the dataset on keypoint_rcnn_R_50_FPN_3x pretrained model from detectron2's model zoo.

Parameters:
datasets and all needed parameters are set during construction of the class and are initialized after calling SingleDemoGraspLearner.fit(self)
function.

#### `SingleDemoGraspLearner.infer`
```python
SingleDemoGraspLearner.infer(self, img_data)
```

Performs inference on a single image. Outputs are status, bounding box and predicted keypoints

Parameters:
- **img**: *object*
  Object of type engine.data.Image.
    
#### `SingleDemoGraspLearner.infer_raw_output`
```python
SingleDemoGraspLearner.infer_raw_output(self, img_data)
```

Performs inference on a single image. Outputs are prediction instances in detectron format.

Parameters:
- **img**: *object*
  Object of type engine.data.Image.



#### `SingleDemoGraspLearner.load`
```python
SingleDemoGraspLearner.load(self, path_to_model)
```
Loads a model from the path provided.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.

#### `SingleDemoGraspLearner.download`
```python
SingleDemoGraspLearner.download(self, path, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:
- **path**: *str, default=None*
  Specifies the folder where pre-trained model will be downloaded. If *None*, the *self.output_dir* directory will be used instead.
  
- **url**: *str, default=OpenDR FTP URL* 
  URL of the FTP server.
 


#### Environment setup
The repository consists of two main parts: First, an augmentation script where a single or multiple views of an object (Image format) are given as input,
and the user will be asked to annotate those images as instructions given during augmentation. Secondly, the learner class for single_demo_grasp module. The learner class depends
on the external package of Detectron2 that must be installed beforehand to run the module properly. 


#### Examples
```python
from SingleDemoGraspLearner import *     # necessary libraries and definitions
 
test_image = cv2.imread("image.png")
learner = SingleDemoGraspLearner() #create an instance of learner class
learner.load("models/piston.pth") #path to pre-trained method and output score threshold
success_flag, bounding_box, keypoints = learner.infer(test_image)
print(bounding_box[0])
print(keypoints[0])
```

#### References
<a name="Detectron2" href="https://github.com/facebookresearch/detectron2">[1]</a> detectron2 github repository,
[Github](https://github.com/facebookresearch/detectron2).
