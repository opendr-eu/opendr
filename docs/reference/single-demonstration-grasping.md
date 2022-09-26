# single_demo_grasp module

The *single_demo_grasp* module contains the *SingleDemoGraspLearner* class, which inherits from the abstract class *LearnerRL*.

### Class SingleDemoGraspLearner
Bases: `engine.learners.LearnerRL`

The *SingleDemoGraspLearner* class is a wrapper class based on [Detectron2](https://github.com/facebookresearch/detectron2) implementation of Keypoint-rcnn keypoint architecture. This module is used to train a keypoint detection module which its predictions are utilized to extract planar grasp poses that are translated to 6D grasp pose. A demonstration is then provided as an example in the projects directory.

The [SingleDemoGraspLearner](/src/opendr/control/single_demo_grasp/training/single_demo_grasp_learner.py) class has the following public methods:

#### `SingleDemoGraspLearner` constructor

Constructor parameters:


- **lr**: *float, default=8e-4*\
  Specifies the learning rate to be used during training.
- **iters**: *int, default=1000*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=512*\
  Specifies the batch size during training for network's ROI_HEADS per image.
- **img_per_step**: *int, default=2*\
  Training images per step per GPU.
- **data_directory**: *str, default=None*\
  Specifies a path where the learner class stores saves checkpoints for each object.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **object_name**: *str, default=None*\
  Name of the object to be used. It should be specified correctly as it is used to find the path to the augmented dataset.
- **num_classes**: *int, default=1*\
  Number of classes depending on how many objects are used for training.
- **num_workers**: *int, default=2*\
  number of data loading workers
- **threshold**: *float, default=8e-1*\
  Specifies the threshold to filter out outputs of inference based on prediction score.

#### `SingleDemoGraspLearner.fit`
```python
SingleDemoGraspLearner.fit(self)
```

Train the agent on the environment. This method does not accept any parameters as they are automatically set duting learner initialization.

#### `SingleDemoGraspLearner.infer`
```python
SingleDemoGraspLearner.infer(self, img_data):
```
Runs inference on a single image frame.

Parameters:

- **img_data**: *opendr.engine.data.Image*\
  Path to save the model, including the filename.

#### `SingleDemoGraspLearner.save`
```python
SingleDemoGraspLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `SingleDemoGraspLearner.load`
```python
SingleDemoGraspLearner.load(self, path_to_model)
```
Loads a model from the path provided.

Parameters:

- **path_to_model**: *str*\
  Path of the model to be loaded.

#### `SingleDemoGraspLearner.download`
```python
SingleDemoGraspLearner.download(self, path, verbose, object_name)
```
Loads a model from the path provided.

Parameters:

- **path**: *str, default=None*\
  Path for downloading content.
- **object_name**: *str, default=None*\
  Name of the object to download its corresponding pretrained model and data.

## Workspace Setup

In order to run the demo, [Webots](https://cyberbotics.com/#download) simulator is required.
- download Webots 2021b for your platform from [here](https://github.com/cyberbotics/webots/releases/tag/R2021b) and install it
- install webots-ros, where ROS_DISTRO must be either `melodic` or `noetic`
```
$ sudo apt-get install ros-ROS_DISTRO-webots-ros
```
- set the environment variable below, by pointing to the location where Webots was installed.
In ubuntu you can do so by executing the following command in a terminal:
```
$ export WEBOTS_HOME=/usr/local/webots
```

From the OpenDR folder, the workspace can be setup by installing the compilation and runtime dependencies:

```
$ make install_compilation_dependencies
$ make install_runtime_dependencies
```

after installing dependencies, the user must source the workspace in the shell in order to detect the packages:

```
$ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
```

## Demos

Three different nodes must be launched consecutively in order to properly run the grasping pipeline:

1. first, the simulation environment must be loaded. open a new terminal and run the following commands:

```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ export WEBOTS_HOME=/usr/local/webots
5. $ roslaunch single_demo_grasping_demo panda_sim.launch
```

2. secondly, open a second terminal and run camera stream node that runs inference on images from camera:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ roslaunch single_demo_grasping_demo camera_stream_inference.launch
```

3. finally, open a third terminal and run commander node to control the robot step by step:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ roslaunch single_demo_grasping_demo panda_sim_control.launch
```

## Example

You can find an example on how to use the learner class to run inference and see the result in the following directory:
```
$ cd projects/python/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
```
simply run:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/python/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ cd projects/python/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
5. $ ./single_demo_inference.py
```

## Performance Evaluation

TABLE-1: OpenDR Single Demonstration Grasping platform inference speeds.
| Platform              | Inference speed (FPS)  |
| --------------------- | ---------------------- | 
| Nvidia GTX 1080 ti    | 20                     |
| Nvidia Geforce 940mx  | 2.5                    | 
| Jetson Xavier NX      | 4                      | 
| CPU                   | 0.4                    | 



The energy consumption of the detection model during inference was also measured on Xavier NX and reported accordingly.
It is worth mentioning that the inference on the first iteration requires more energy for initialization which as it can be seen in TABLE-2.

TABLE-2: OpenDR Single Demonstration Grasping energy consumptions and memory usage.
| Stage                       | Energy (Joules)  |
| --------------------------- | ---------------- | 
| First step (initialization) | 12               |
| Normal                      | 3.4              | 


TABLE-3: OpenDR Single Demonstration Grasping training.
|   Model        | Dataset size                      | Training Time <br> (hr:min:sec) | Model size (MB)               | 
|--------------- |---------------------------------- |-------------------------------- |------------------------------ |
| A              | Faster R-CNN: 1500 <br> CNN: 5000 | 00:14:00 <br> 00:02:00          | Faster R-CNN: 300 <br> CNN: 8 |              
| B              | 1500                              | 00:07:30                        | 450                           |                              
| C (simulation) | 1500                              | 00:07:00                        | 450                           | 


TABLE-4: OpenDR Single Demonstration Grasping inferences success evaluation. 
|   Model        | Success rate  |
|--------------- |-------------- |
| A              | 0.913         |
| B              | 0.825         |
| C (simulation) | 0.935         | 


Finally, we evaluated the ability of the provided tool to run on different platforms.
The tool has been verified to run correctly on the platforms reported in Table TABLE-5. 

TABLE-5: OpenDR Single Demonstration Grasping platform compatibility evaluation.
| Platform                                     | Test results           |
| -------------------------------------------- | ---------------------- | 
| x86 - Ubuntu 20.04 (bare installation - CPU) | Pass                   |
| x86 - Ubuntu 20.04 (bare installation - GPU) | Pass                   | 
| x86 - Ubuntu 20.04 (pip installation)        | Not supported          | 
| x86 - Ubuntu 20.04 (CPU docker)              | Pass*                  |  
| x86 - Ubuntu 20.04 (GPU docker)              | Pass*                  | 
| NVIDIA Jetson TX2                            | Not tested             |
| NVIDIA Jetson Xavier AGX                     | Not tested             |
| NVIDIA Jetson Xavier NX                      | Pass**                 |

\* Installation only considers the learner class. For running the simulation, extra steps are required. \*\* The installation script did not include detectron2 module and webots installation which had to be installed manually with slight modifications and building the detectron2 from source as there was no prebuilt wheel for aarch64 architecture.
