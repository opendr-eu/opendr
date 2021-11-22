## synthetic_facial_image_generator module

The *synthetic_facial_image_generator* module contains the *MultiviewDataGenerationLearner* class, which inherits from the abstract class *Learner*.

### Class MultiviewDataGenerationLearner
Bases: `engine.learners.Learner`

The *MultiviewDataGenerationLearner* class is a wrapper of the Rotate-and-Render [[1]](#R-R-paper) photorealistic multi-view facial image generator based on the original
[Rotate-and-Render implementation](https://github.com/Hangz-nju-cuhk/Rotate-and-Render).
It can be used to perform multi-view facial image generation from a single view image on the wild (eval). 
The [MultiviewDataGenerationLearner](#projects.data_generation.synthetic-multi-view-facial-image-generation.3ddfa.SyntheticDataGeneration.py ) class has the
following public methods:

#### `MultiviewDataGenerationLearner` constructor
```python
MultiviewDataGenerationLearner(self, path_in='./example/Images', path_3ddfa='./', save_path='./results', val_yaw='10,20', val_pitch=' 30,40', device='cuda')
```

Constructor parameter explanation:
- **path_in**: *str, default='./example/Images'* \
An absolute path (path in) which indicates the folder that contains the set of single view facial image snapshots to be processed by the algorithm.
- **path_3ddfa**: *str, default='./'* \
An absolute path (path 3ddfa) which indicates the 3ddfa module folder of the software structure as presented in the repository. This path is necessary in order 
for the software to create the folders for the intermediate / temporary storage of files generated during the pre-processing such as 3d face models, facial landmarks etc 
in the folder results of this path.
- **save_path**: *str, default='./results'* \
The output images are stored in the folder indicated by save path which is also a class input parameter.
- **val_yaw**: *str, default='10,20'* \
Definition of the yaw angles (in the interval [−90◦,90◦]) for which the rendered images will be produced.
- **val_pitch**: *str, default=' 30,40'* \
Definition of the pitch angles (in the interval [−90◦,90◦]) for which the rendered images will be produced.
- **device**: *{'cuda', 'cpu'}, default='cpu'* \
Specifies the device to be used.


#### `MultiviewDataGenerationLearner.eval`
```python
MultiviewDataGenerationLearner.eval()
```

This function is implementing the main procedure for the creation of the multi-view
facial images, which consists of three different stages. Instead of initializing the main
parameters of the 3DDFA network in the intializer , the first stage includes detection of the
candidate faces in the input images and 3D-head mesh fitting using 3DDFA. Moreover,
the second stage extracts the facial landmarks in order to derive the head pose and align
the images with the 3d head model mesh. Finally, the main functionality of the multiview
facial image rendering is executed by loading the respective network parameters.

### Usage Example

```python
import path_helper
import argparse
from SyntheticDataGeneration import MultiviewDataGenerationLearner
__all__ = ['path_helper']

parser=argparse.ArgumentParser()
parser.add_argument('-path_in', default='/home/user/Pictures/', type=str )
parser.add_argument('-save_path', default='./results', type=str )
parser.add_argument('-path_3ddfa', default='/opendr_internal/projects/data_generation/synthetic_multi_view_facial_image_generation/DDFA', type=str)
parser.add_argument('-val_yaw',  default='10,20', nargs='+',type=str)
parser.add_argument('-val_pitch', default='30,40', nargs='+', type=str)
parser.add_argument('-device', default='cuda', type=str)
args=parser.parse_args()
synthetic = MultiviewDataGenerationLearner(path_in=args.path_in, path_3ddfa=args.path_3ddfa, save_path=args.save_path, val_yaw=args.val_yaw, val_pitch=args.val_pitch, device=args.device)
synthetic.eval()
```
The corresponding paths for the input, output folders as well as the pitch and yaw angles for which the user wants to
produce the facial images can be easily incorporated in the class creation while the method is initialized. 
The process is executed for the CNN parameters and GPUs specified in the code. Users that wish to modify these parameters shall change the respective code
A parser is created and the arguments path in, path_3ddfa, save_path, val_yaw, val_pitch which were described above are determined. Subsequently, an object synthetic
of the class ```MultiviewDataGenerationLearner``` is created and the function ```synthetic.eval()``` is executed.

#### References
<a name="R-R-paper" href="https://github.com/Hangz-nju-cuhk/Rotate-and-Render">[1]</a>
Hang Zhou, Jihao Liu, Ziwei Liu, Yu Liu, Xiaogang Wang, Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images,
[arXiv](https://arxiv.org/abs/2003.08124#).  
