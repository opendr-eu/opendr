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

- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device to be used.

- **checkpoint_dir**: *str, default='utilities/PIFu/checkpoints'*
  Specifies a path to be used for loading the checkpoints for inference. 
  
#### `PIFuGeneratorLearner.infer`
```python
PIFuGeneratorLearner.infer(self, mgs_rgb, imgs_msk, obj_path, extract_pose)
```
  
#### `PIFuGeneratorLearner.load`
```python
PIFuGeneratorLearner.load(self, path)
```  
#### `PIFuGeneratorLearner.download`
```python
PIFuGeneratorLearner.download(self, path, url)
```  

#### `PIFuGeneratorLearner.get_img_views`
```python
PIFuGeneratorLearner.get_img_views(self, model_3D, rotations, human_pose_3D, plot_kps)
```  
  
#### References
<a name="pifu-paper" href="https://shunsukesaito.github.io/PIFu/">[1]</a>
PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,
[arXiv](https://arxiv.org/abs/1905.05172).  
