# end_to_end_planning module

The *end_to_end_planning* module contains the *EndToEndPlanningRLLearner* class, which inherits from the abstract 
class *LearnerRL*.

### Class EndToEndPlanningRLLearner
Bases: `engine.learners.LearnerRL`

The *EndToEndPlanningRLLearner* is an agent that can be used to train quadrotor robots equipped with a depth sensor to
follow a provided trajectory while avoiding obstacles. Originally published in [[1]](#safe-e2e-planning),

The [EndToEndPlanningRLLearner](../../src/opendr/planning/end_to_end_planning/e2e_planning_learner.py) class has the 
following public methods:

#### `EndToEndPlanningRLLearner` constructor

Constructor parameters:

- **env**: *gym.Env, default=None*\
  Reinforcement learning environment to train or evaluate the agent on.
- **lr**: *float, default=3e-4*\
  Specifies the initial learning rate to be used during training.
- **n_steps**: *int, default=1024*\
  Specifies the number of steps to run for environment per update.
- **iters**: *int, default=1e5*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*\
  Specifies the batch size during training.
- **checkpoint_after_iter**: *int, default=500*\
  Specifies per how many training steps a checkpoint should be saved.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm stores log files and saves checkpoints.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.

#### `EndToEndPlanningRLLearner.fit`
```python
EndToEndPlanningRLLearner.fit(self, env, logging_path, verbose)
```

Train the agent on the environment.

Parameters:

- **env**: *gym.Env, default=None*\
  If specified use this env to train.
- **logging_path**: *str, default=''*\
  Path for logging and checkpointing.
- **verbose**: *bool, default=True*\
  Enable verbosity.


#### `EndToEndPlanningRLLearner.eval`
```python
EndToEndPlanningRLLearner.eval(self, env)
```
Evaluate the agent on the specified environment.

Parameters:

- **env**: *gym.Env, default=None*\
  Environment to evaluate on.


#### `EndToEndPlanningRLLearner.save`
```python
EndToEndPlanningRLLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `EndToEndPlanningRLLearner.load`
```python
EndToEndPlanningRLLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.


#### `EndToEndPlanningRLLearner.infer`
```python
EndToEndPlanningRLLearner.infer(self, batch, deterministic)
```
Performs inference on a single observation or a list of observations.

Parameters:

- **batch**: *dict or list of dict, default=None*\
  Single observation or list of observations.
- **deterministic**: *bool, default=True*\
  Use deterministic actions from the policy

### Simulation environment setup

The environment is provided with a [world](../../src/opendr/planning/end_to_end_planning/envs/webots/worlds/train-no-dynamic-random-obstacles.wbt)
that needs to be opened with Webots version 2022b in order to demonstrate the end-to-end planner.

The environment includes an optional Ardupilot controlled quadrotor for simulating dynamics. 
For the installation of Ardupilot instructions are available [here](https://github.com/ArduPilot/ardupilot).

The required files to complete Ardupilot setup can be downloaded by running [download_ardupilot_files.py](../../src/opendr/planning/end_to_end_planning/download_ardupilot_files.py) script.
The downloaded files (zipped as `ardupilot.zip`) should be replaced under the installation of Ardupilot.
In order to run Ardupilot in Webots 2021a, controller codes should be replaced. (For older versions of Webots, these files can be skipped.)
The world file for the environment is provided under `/ardupilot/libraries/SITL/examples/webots/worlds/` for training and testing.

Install `mavros` package for ROS communication with Ardupilot.
Instructions are available [here](https://github.com/mavlink/mavros/blob/master/mavros/README.md#installation).
Source installation is recommended.

### Running the environment

The following steps should be executed to have a ROS communication between Gym environment and simulation.
- Start the Webots and open the provided world file. 
The simulation time should stop at first time step and wait for Ardupilot software to run.
- Run following script from Ardupilot directory: `./libraries/SITL/examples/Webots/dronePlus.sh` which starts software in the loop execution of the Ardupilot software.
- Run `roscore`.
- Run `roslaunch mavros apm.launch` which creates ROS communication for Ardupilot.
- Run following ROS nodes in `src/opendr/planning/end_to_end_planning/src`:
  - `children_robot` which activates required sensors on quadrotor and creates ROS communication for them. 
  - `take_off` which takes off the quadrotor.
  - `range_image` which converts the depth image into array format to be input for the learner.
  
After these steps the [UAVDepthPlanningEnv](../../src/opendr/planning/end_to_end_planning/envs/UAV_depth_planning_env.py) gym environment can send action comments to the simulated drone and receive depth image and pose information from simulation. 

### Examples

Training in Webots environment:

```python
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner, UAVDepthPlanningEnv

env = UAVDepthPlanningEnv()
learner = EndToEndPlanningRLLearner(env, n_steps=1024)
learner.fit(logging_path='./end_to_end_planning_tmp')
```


Running a pretrained model:

```python
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner, UAVDepthPlanningEnv

env = UAVDepthPlanningEnv()
learner = EndToEndPlanningRLLearner(env)
learner.load('{$OPENDR_HOME}/src/opendr/planning/end_to_end_planning/pretrained_model/saved_model.zip')
obs = env.reset()
sum_of_rew = 0
number_of_timesteps = 20
for i in range(number_of_timesteps):
    action, _states = learner.infer(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    sum_of_rew += rewards
    if dones:
        obs = env.reset()
print("Reward collected is:", sum_of_rew)
```

### Performance Evaluation

TABLE 1: Speed (FPS) and energy consumption for inference on various platforms.

|                 | TX2   | Xavier | RTX 2080 Ti |
| --------------- | ----- | ------ | ----------- |
| FPS Evaluation  | 153.5 | 201.6  | 973.6       |
| Energy (Joules) | 0.12  | 0.051  | \-          |

TABLE 2: Platform compatibility evaluation.

| Platform                                     | Test results |
| -------------------------------------------- | ------------ |
| x86 - Ubuntu 20.04 (bare installation - CPU) | Pass         |
| x86 - Ubuntu 20.04 (bare installation - GPU) | Pass         |
| x86 - Ubuntu 20.04 (pip installation)        | Pass         |
| x86 - Ubuntu 20.04 (CPU docker)              | Pass         |
| x86 - Ubuntu 20.04 (GPU docker)              | Pass         |
| NVIDIA Jetson TX2                            | Pass         |
| NVIDIA Jetson Xavier AGX                     | Pass         |

#### References
<a name="safe-e2e-planning" href="https://github.com/open-airlab/gym-depth-planning.git">[1]</a> Ugurlu, H.I.; Pham, X.H.; Kayacan, E. Sim-to-Real Deep Reinforcement Learning for Safe End-to-End Planning of Aerial Robots. Robotics 2022, 11, 109. 
[DOI](https://doi.org/10.3390/robotics11050109). [GitHub](https://github.com/open-airlab/gym-depth-planning.git)