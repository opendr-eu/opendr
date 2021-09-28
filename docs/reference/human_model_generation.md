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
PIFuGeneratorLearner(self, model_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter,
temp_path, device, threshold, num_classes, return_segmentations)
```
#### References
<a name="pifu-paper" href="https://shunsukesaito.github.io/PIFu/">[1]</a>
PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,
[arXiv](https://arxiv.org/abs/1905.05172).  
