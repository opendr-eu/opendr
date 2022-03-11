# end_to_end_planning module

The *end_to_end_planning* module contains the *EndToEndPlanningRLLearner* class, which inherits from the abstract 
class *LearnerRL*.

### Class EndToEndPlanningRLLearner
Bases: `engine.learners.LearnerRL`



The [EndToEndPlanningRLLearner](/src/opendr/planning/end_to_end_planning/e2e_planning_learner.py) class has the 
following public methods:

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




### Simulation environment setup

The environment includes an Ardupilot controlled quadrotor in Webots simulation. 
Installation of Ardupilot software follows this link: `https://github.com/ArduPilot/ardupilot`

The files under `ardupilot.zip` should be replaced under the installation of Ardupilot, can be downloaded by running 
`download_ardupilot_files.py` script.
In order to run the Ardupilot in Webots 2021a controller codes should be replaced. 
For older versions of Webots, these files can be skipped.
The world file for the environment is provided 
under `/ardupilot/libraries/SITL/examples/webots/worlds/` for training and testing.

Install `mavros` package into catkin workspace for ROS communication with Ardupilot. 

### Running the environment

The following steps should be executed to have a ROS communication between Gym environment and simulation.
- Start the Webots and open the provided world file. The simulation time should stop at first time step and wait for 
Ardupilot software to run.
- Run following script from Ardupilot directory: `./libraries/SITL/examples/Webots/dronePlus.sh` which starts 
software in the loop execution of the Ardupilot software.
- Run `roscore`.
- Run `roslaunch mavros apm.launch` which creates ROS communication for Ardupilot.
- Run following ROS nodes in `src/opendr/planning/end_to_end_planning/src`:
  - `children_robot` which activates required sensors on quadrotor and creates ROS communication for them. 
  - `take_off` which takes off the quadrotor.
  - `range_image` which converts the depth image into array format to be input for the learner.
  
After these steps the [AgiEnv](src/opendr/planning/end_to_end_planning/envs/agi_env.py) gym environment can send 
action comments to the simulated drone and receive depth image and pose information from simulation. 

### Examples

Training in Webots environment:

```python
from opendr.planning.end_to_end_planning.e2e_planning_learner import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.envs.agi_env import AgiEnv

env = AgiEnv()
learner = EndToEndPlanningRLLearner(env, n_steps=1024)
learner.fit(logging_path='./end_to_end_planning_tmp')
```


Running a pretrained model:

```python
from opendr.planning.end_to_end_planning.e2e_planning_learner import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.envs.agi_env import AgiEnv

env = AgiEnv()
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