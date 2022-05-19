## synthetic_facial_image_generator module

The *synthetic_facial_image_generator* module contains the *MultiviewDataGeneration* class, which implements the multi-view facial image rendering operation.

### Class MultiviewDataGeneration

The *MultiviewDataGeneration* class is a wrapper of the Rotate-and-Render [[1]](#R-R-paper) photorealistic multi-view facial image generator based on the original
[Rotate-and-Render implementation](https://github.com/Hangz-nju-cuhk/Rotate-and-Render).
It can be used to perform multi-view facial image generation from a single view image on the wild (eval). 
The [MultiviewDataGeneration](#projects.data_generation.synthetic-multi-view-facial-image-generation.3ddfa.SyntheticDataGeneration.py ) class has the
following public methods:

#### `MultiviewDataGeneration` constructor
```python
MultiviewDataGeneration(self, args)
```

Constructor main parameters *args* explanation:

- **path_in**: *str, default='./example/Images'* \
An absolute path (path in) which indicates the folder that contains the set of single view facial image snapshots to be processed by the algorithm.
- **path_3ddfa**: *str, default='./'* \
An absolute path (path 3ddfa) which indicates the 3ddfa module folder of the software structure as presented in the repository. This path is necessary in order for the software to create the folders for the intermediate / temporary storage of files generated during the pre-processing such as 3d face models, facial landmarks etc.
in the folder results of this path.
- **save_path**: *str, default='./results'* \
The output images are stored in the folder indicated by save path which is also a class input parameter.
- **val_yaw**: *str, default='10,20'* \
Definition of the yaw angles (in the interval [−90°,90°]) for which the rendered images will be produced.
- **val_pitch**: *str, default=' 30,40'* \
Definition of the pitch angles (in the interval [−90°,90°]) for which the rendered images will be produced.
- **device**: *{'cuda', 'cpu'}, default='cpu'* \
Specifies the device to be used.


#### `MultiviewDataGeneration.eval`
```python
MultiviewDataGeneration.eval()
```

This function is implementing the main procedure for the creation of the multi-view facial images, which consists of three different stages.
Instead of initializing the main parameters of the 3DDFA network in the intializer, the first stage includes detection of the candidate faces in the input images and 3D-head mesh fitting using 3DDFA.
Moreover, the second stage extracts the facial landmarks in order to derive the head pose and align the images with the 3d head model mesh.
Finally, the main functionality of the multiview facial image rendering is executed by loading the respective network parameters.

### Usage Example

```python
python3 tool_synthetic_facial_generation.py -path_in ./demos/imgs_input/ -path_3ddfa ./algorithm/DDFA/ -save_path ./results -val_yaw 10, 40 -val_pitch 10, 30 -device cuda
```
The corresponding paths for the input, output folders as well as the pitch and yaw angles for which the user wants to
produce the facial images can be easily incorporated in the class creation while the method is initialized. 
The process is executed for the CNN parameters and GPUs specified in the arguments of the aforementioned command.
Users that wish to modify these parameters shall change the respective input arguments which derived from a parser including the arguments path in, path_3ddfa, save_path, val_yaw, val_pitch etc. 

### Performance Evaluation

In this subsection, we measure the inference speed of the synthetic multi-view facial image generation algorithm integrated in the OpenDR toolkit.
The results are presented in the following Table.
These tests have been performed using a GPU NVIDIA RTX 2070 and the reported time concerns one sample facial image.
Note that the integrated tool currently only supports GPU inference. 


| Method                                      |   GPU NVIDIA RTX 2070 (ms)| 
|---------------------------------------------|---------------------------|
|Synthetic Multi-view Facial Image Generation |      20.41                | 

In the following Table the results of running the Synthetic Multi-view Facial Image Generation tool for different platforms and varying installation procedures are provided.
Again, note that this tool mainly targets offline data generation.
Therefore, it has not been tested on embedded platforms, since they are not usually used for generating data for training.

| Platform                                     | Compatibility  |
|----------------------------------------------|----------------|
| x86 - Ubuntu 20.04 (bare installation - CPU) | ❌             |
| x86 - Ubuntu 20.04 (bare installation - GPU) | ✔️             |
| x86 - Ubuntu 20.04 (pip installation)        | ✔️             |
| x86 - Ubuntu 20.04 (CPU docker)              | ❌             |
| x86 - Ubuntu 20.04 (GPU docker)              | ✔️             |
| NVIDIA Jetson Xavier NX                      | N/A            |
| NVIDIA Jetson TX2                            | N/A            |

#### References
<a name="R-R-paper" href="https://github.com/Hangz-nju-cuhk/Rotate-and-Render">[1]</a>
Hang Zhou, Jihao Liu, Ziwei Liu, Yu Liu, Xiaogang Wang, Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images,
[arXiv](https://arxiv.org/abs/2003.08124#).  
