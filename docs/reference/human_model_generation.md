## human_model_generation module

The *human_model_generation* module contains the *pifu_generator_learner.py* class, which inherits from the abstract class *Learner*.

### Class Pifu_generator_learner
Bases: `engine.learners.Learner`

The *Pifu_generator_learner* class is a wrapper of the PIFu [[1]](#pifu-paper) object detection algorithm based on the original
[PIFu implementation](https://github.com/shunsukesaito/PIFu).
It can be used to perform human model generation from a single image(inference). In addtion, the *Pifu_generator_learner* enables the 3D pose approximation of a generated human model as well as the generation of multi-view renderings of the human model.

The [Pifu_generator_learner](#src.opendr.simulation.human_model_generation.pifu_generator_learner.py ) class has the
following public methods:


#### References
<a name="pifu-paper" href="https://shunsukesaito.github.io/PIFu/">[1]</a>
PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization,
[arXiv](https://arxiv.org/abs/1905.05172).  
