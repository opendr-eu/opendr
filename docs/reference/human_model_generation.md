## human_model_generation module

The *human_model_generation* module contains the *PIFuGeneratorLearner* class, which inherits from the abstract class *Learner*.

### Class PIFuGeneratorLearner
Bases: `engine.learners.Learner`

The *PIFuGeneratorLearner* class is a wrapper of the PIFu [[1]](#pifu-paper) object detection algorithm based on the original
[PIFu implementation](https://github.com/shunsukesaito/PIFu).
It can be used to perform human model generation from a single image(inference). In addtion, the *PIFuGeneratorLearner* enables the 3D pose approximation of a generated human model as well as the generation of multi-view renderings of the human model.

The [PIFuGeneratorLearner](#src.opendr.simulation.human_model_generation.pifu_generator_learner.py ) class has the
following public methods:

#### `PIFuGeneratorLearner` constructor
```python
PIFuGeneratorLearner(self, device, checkpoint_dir)
```

Constructor parameter explanation:
- **device**: *{'cuda', 'cpu'}, default='cuda'*\
Specifies the device to be used.
- **checkpoint_dir**: *str, default='utilities/PIFu/checkpoints'*\
Specifies a path to be used for loading the checkpoints for inference. 
  
#### `PIFuGeneratorLearner.infer`
```python
PIFuGeneratorLearner.infer(self, imgs_rgb, imgs_msk, obj_path, extract_pose)
```

This method generates a 3D human model from a single image. The joints of the 3D model in the 3D space can be optionally approximated. A future release will allow the use of multiple images as input. 

Parameters:
- **imgs_rgb**: *list, default=None*\
List of images of type engine.data.Image. Those images will be used as input. At the current release, the list's length must be 1. 
- **imgs_msk**: *list, default=None*\
List of images of type engine.data.Image. Those images will be used as masks, depicting the silhouette of the portrayed human. At the current release, the list's length must be 1. 
- **obj_path**: *str, default=None*\
Specifies a path for saving the generated 3D human model in OBJ format.
 - **extract_pose**: *bool, default=False*\
Specifies whether the joints of the 3D model in the 3D space will be approximated or not.

#### `PIFuGeneratorLearner.load`
```python
PIFuGeneratorLearner.load(self, path)
```  

This method loads a pretrained model.

Parameters:
- **path**: *str*\
Specifies the folder where the model will be loaded from.
  
#### `PIFuGeneratorLearner.download`
```python
PIFuGeneratorLearner.download(self, path, url)
```  

This method downloads the needed pretrained models.

Parameters:
- **path**: *str, default=None*\
Specifies the folder where data will be downloaded.
  
- **url**: *str, default=OpenDR FTP URL*\
URL of the FTP server.

#### `PIFuGeneratorLearner.get_img_views`
```python
PIFuGeneratorLearner.get_img_views(self, model_3D, rotations, human_pose_3D, plot_kps)
```  
This method generate renderings of the generated 3D human model from a given list of rotation angles (yaw only).

Parameters:
- **model_3D**: *object, default=None*\
Object of type simulation.human_model_genration.utilities.model_3D.Model_3D. It holds the human 3D model. 

- **rotations**: *list, default=None*\
List of yaw angles used in the generation of the renderings of a 3D human model from various views.

- **human_pose_3D**: *list, default=None*\
List of keypoints, which contains the name of each keypoint along with their coodinates [x,y,z] in the 3D space.

- **plot_kps**: *bool, default=False*\ 
Specifies whether the projections of the joints will be plotted in the renderings.

  
#### References
<a name="pifu-paper" href="https://shunsukesaito.github.io/PIFu/">[1]</a>
PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,
[arXiv](https://arxiv.org/abs/1905.05172).  
