# mobile_manipulation module

The *mobile_manipulation* module contains the *MobileRLLearner* class, which inherits from the abstract class *RLLearner*.

### Class MobileRLLearner
Bases: `engine.learners.LearnerRL`



The *MobileRLLearner* class is an RL agent that can be used to train wheeled robots for mobile manipulation in conjunction with the 
environments. Originally developed in [[1]](#kinematic-feasibility) implementation found on [Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning](http://kinematic-rl.cs.uni-freiburg.de/).

The [MobileRLLearner](#src.control.mobile_manipulation.mobile_manipulation_learner.py) class has the
following public methods:

#### `MobileRLLearner` constructor

Constructor parameters:
- **lr**: *float, default=1e-5*  
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=1'000'000*  
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*  
  Specifies the batch size during training.
- **lr_schedule**: *{'', 'linear'}, default='linear'*  
  Specifies the learning rate scheduler to use. Empty to use a constant rate.
- **lr_end**: *float, default=1e-6*
  Specifies the final learning rate if a lr_schedule is used.
- **backbone**: *{'MlpPolicy'}, default='MlpPolicy'*  
  Specifies the architecture for the RL agent.
- **checkpoint_after_iter**: *int, default=20_000*  
  Specifies per how many training steps a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default='temp'*  
  Specifies a path where the algorithm looks for checkpoints, the checkpoints are saved along with the logging files and configuration.
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **seed**: *int, default=None*  
  Random seed for the agent. If None a random seed will be used.
- **buffer_size**: *int, default=100_000*  
  Size of the replay buffer.
- **learning_starts**: *int, default=0*  
  Number of environment steps with a random policy before starting training.
- **tau**: *float, default=0.001*  
  Polyak averaging of the target network.
- **gamma**: *float, default=0.99*  
  Discount factor.
- **explore_noise**: *float, default=0.5*  
  Strength of the exploration noise.
- **explore_noise_type**: *{'normal', 'OU'}, default=normal*  
  Type of exploration noise, either normal or Ornstein Uhlenbeck (OU) noise.
- **ent_coef**: *{'auto', int}, default='auto'*  
  Entropy coefficient for SAC, 'auto' to learn the coefficient.
- **nr_evaluations**: *int, default=50*  
  Number of episodes to evaluate over.
- **evaluation_frequency**: *int, default=20_000*  
  Number of steps after which to episodically evaluate during training.

#### `MobileRLLearner.fit`
```python
MobileRLLearner.fit(self, env=None, val_env=None, logging_path='', silent=True, verbose=True)
```

Train the agent on the environment.

Parameters:
- **env**: *gym.Env, default=None*
  If specified use this env to train.
- **val_env**:  *gym.Env, default=None*
  If specified periodically evaluate on this env.
- **logging_path**: *str, default = ''*
  Path for logging and checkpointing.
- **silent**: *bool, default=False* 
  Disable verbosity.
- **verbose**: *bool, default=True*
  Enable verbosity.


#### `MobileRLLearner.eval`
```python
MobileRLLearner.eval(self, env, name_prefix='', nr_evaluations: int = None)
```
Evaluate the agent on the specified environment.

Parameters:
- **env**: *gym.Env, default=None*
  Env to evaluate on.
- **name_prefix**: *str, default=''*
  Name prefix for all logged variables.
- **nr_evaluations**: *int, default=None*
  Number of episodes to evaluate over.
  

#### `MobileRLLearner.save`
```python
MobileRLLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:
- **path**: *str*  
  Path to save the model, including the filename.


#### `MobileRLLearner.load`
```python
MobileRLLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.


#### ROS Setup
The repository consists of two main parts: a training environment written in C++ and connected to python through bindings and the RL agents written in python3.

As not all ROS packages work with python3, the setup relies on running the robot-specific packages in a python2 environment
and our package in a python3 environment.
The environment was tested for Ubunto 18.04 and ROS melodic.

###### Installation
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings

    sudo apt install ros-[version]-pybind11-catkin

We provide implementations for the PR2, PAL Tiago and Toyota HSR robots. The following outlines the installation for the PR2.
For the other robots please follow the official guides to install the respective ROS dependencies. We recommend to use
separate catkin workspaces for each.

Install moveit and the pr2

    sudo apt-get install ros-[version]-moveit
    sudo apt-get install ros-[version]-pr2-simulator
    sudo apt-get install ros-[version]-moveit-pr2

Install libgp: `https://github.com/mblum/libgp.git`

Create a catkin workspace (ideally a separate one for each robot)

    mkdir ~/catkin_ws
    cd catkin_ws

Fork the repo and clone into `./src`

    cd src
    git clone [url] src/modulation_rl

Create a python environment. We recommend using conda, which requires to first install Anaconda or Miniconda. Then do

    conda env create -f src/modulation_rl/environment.yml
    conda activate modulation_rl

Configure the workspace to use your environment's python3 (adjust path according to your version)

    catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

Build the workspace

    catkin build

Each new build of the ROS / C++package requires a

    source devel/setup.bash

To be able to visualise install rviz

    http://wiki.ros.org/rviz/UserGuide


##### Run
1. start a roscore

        roscore

2a. training or evaluation in the analytical environment only:

        roslaunch modulation_rl pr2_analytical.launch
   
2b. evaluation in gazebo: _instead_ of 2a start gazebo with the pr2 robot as well moveit

        roslaunch pr2_gazebo pr2_empty_world.launch
        roslaunch pr2_moveit_config move_group.launch

4. Run the demo script

        python mobile_manipulation_demo.py

5. [Visualisation] start rviz:

        rviz -d rviz_config.config

5b. For HSR / Tiago: adjust the reference frame in rviz from `odom` to `odom combined`

#### Examples
* **Training in the analytical environment and evaluation in Gazebo on the Pick & Place task**.

#### Notes

##### HSR
The HSR environment relies on packages that are part of the proprietory HSR simulator. If you have an HSR account with Toyota,
please follow these steps to use the environment. Otherwise ignore this section to use the other environments we provide.

- Check the commented out parts in the `# HSR` section as well as the building of the workspace further below in the `Dockerfile` to install the requirements.
- Comment in the following lines in `CMakeLists.txt`:

      add_library(dynamic_system_hsr src/dynamic_system_hsr.cpp)
      target_link_libraries(dynamic_system_hsr modulation modulation_ellipses utils ${catkin_LIBRARIES})

  and add them to `pybind_add_module()` and `target_link_libraries()` two lines below that.
- Comment in the hsr parts in `src/pybindings` and the import of HSREnv in `scripts/modulation/envs/modulationEnv.py` to create the python bindings
- Now either build the Dockerfile or rebuild a local workspace (we recommend to use separate catkin workspaces for each robot)


#### References
<a name="kinematic-feasibility" href="https://arxiv.org/abs/2101.05325">[1]</a> Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning,
[arXiv](https://arxiv.org/abs/2101.05325).
