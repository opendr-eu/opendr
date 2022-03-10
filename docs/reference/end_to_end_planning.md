# end_to_end_planning module

The *end_to_end_planning* module contains the *EndToEndPlanningRLLearner* class, which inherits from the abstract class *LearnerRL*.

### Class EndToEndPlanningRLLearner
Bases: `engine.learners.LearnerRL`



The [EndToEndPlanningRLLearner](/src/opendr/planning/end_to_end_planning/e2e_planning_learner.py) class has the following public methods:

#### `EndToEndPlanningRLLearner` constructor

Constructor parameters:

- **env**: *gym.Env*\
  Reinforcment learning environment to train or evaluate the agent on.
- **lr**: *float, default=3e-4*\
  Specifies the initial learning rate to be used during training.
- **n_steps**: *int, default=1024*\
  Specifies the number of steps to run for environment per update.
- **iters**: *int, default=5e4*\
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
EndToEndPlanningRLLearner.fit(self, env, logging_path, silent, verbose)
```

Train the agent on the environment.

Parameters:

- **env**: *gym.Env, default=None*\
  If specified use this env to train.
- **logging_path**: *str, default=''*\
  Path for logging and checkpointing.
- **silent**: *bool, default=False*\
  Disable verbosity.
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




## Simulation RL environment setup

The environment includes an Ardupilot controlled quadrotor in Webots simulation. 
Installation of Ardupilot software follows this link: `https://github.com/ArduPilot/ardupilot`

The files under `ardupilot.zip` should be replaced under the installation of Ardupilot, can be downloaded by running `download_ardupilot_files.py` script.
In order to run the Ardupilot in Webots 2021a controller codes should be replaced. 
For older versions of Webots, these files can be skipped.
The world file for the environment is provided 
under `/ardupilot/libraries/SITL/examples/webots/worlds/Agri2021a.wbt`.

Install `mavros` package into catkin workspace for ROS communication with Ardupilot. 

## Running the environment

The following steps should be executed to have a ROS communication between Gym environment and simulation.
- Start the Webots and run the world file. The simulation time should stop at first time step and wait for Ardupilot software to run.
- Run following script from Ardupilot directory: `./libraries/SITL/examples/Webots/dronePlus.sh`
- Following ROS commands should be executed:
  - `roslaunch mavros apm.launch`
  - Run `children_robot` node
  - Run `take_off` node
  - Run `range_image` node

After these steps the gym environment can send action comments to the 
simulated drone and receive depth image and pose information from simulation. 

