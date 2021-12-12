# single_demo_grasp module

The *single_demo_grasp* module contains the *SingleDemoGraspLearner* class, which inherits from the abstract class *LearnerRL*.

### Class SingleDemoGraspLearner
Bases: `engine.learners.LearnerRL`

The *SingleDemoGraspLearner* class is a wrapper class based on [Detectron2](https://github.com/facebookresearch/detectron2) implementation of Keypoint-rcnn keypoint architecture. This module is used to train a keypoint detection module which its predictions are utilized to extract planar grasp poses that are translated to 6D grasp pose. A demonstration is then provided as an example in projects directory.

The [SingleDemoGraspLearner](#src.opendr.control.single_demo_grasp.training.single_demo_grasp_learner.py) class has the following public methods:

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
- **num_workers**: *int, default=1*\
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

- **path**: *str*\
  Path of the model to be loaded.

#### `SingleDemoGraspLearner.download`
```python
SingleDemoGraspLearner.download(self, path = None, verbose = False, object_name = None)
```
Loads a model from the path provided.

Parameters:

- **path**: *str, default=None*\
  Path for downloading content.
- **object_name**: *str, default=None*\
  Name of the object to download its corresponding pretrained model and data.
  
#### workspace Setup
The workspace will be setup by installing compilation and runtime dependencies when setting up the OpenDR toolkit:

```
$ make install_compilation_dependencies
$ make install_runtime_dependencies
```

after installing dependencies, the user must source the workspace in the shell in order to detect the packages:

```
$ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
```

also, the user need to set the environment variable below to find webots directory:
```
$ export WEBOTS_HOME=/usr/local/webots
```

#### Demos

three different nodes must be launched consecutively in order to properly run the grasping pipeline:

1. first, the simulation environment must be loaded. open a new terminal and run the following commands:

```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ export WEBOTS_HOME=/usr/local/webots
5. $ roslaunch single_demo_grasping_demo panda_sim.launch 
```

2. secondly, open a second terminal and run camera stream node that runs inference on images from camera:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ roslaunch single_demo_grasping_demo camera_stream_inference.launch.launch 
```

3. finally, open a third terminal and run commander node to control the robot step by step:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash 
4. $ roslaunch single_demo_grasping_demo panda_sim_control.launch 
```

#### Examples
You can find an example on how to use the learner class to run inference and see the result in the following directory:
```
$ cd projects/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/
```
simply run:
```
1. $ cd path/to/opendr/home # change accordingly
2. $ source bin/setup.bash
3. $ source projects/control/single_demo_grasp/simulation_ws/devel/setup.bash
4. $ cd projects/control/single_demo_grasp/simulation_ws/src/single_demo_grasping_demo/inference/ 
5. $ ./single_demo_inference.py
```


