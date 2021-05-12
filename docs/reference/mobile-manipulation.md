# mobile_manipulation module

The *mobile_manipulation* module contains the *MobileRLLearner* class, which inherits from the abstract class *LearnerRL*.

### Class MobileRLLearner
Bases: `engine.learners.LearnerRL`

The *MobileRLLearner* class is an RL agent that can be used to train wheeled robots for mobile manipulation in conjunction with the
`mobileRL.env.mobile_manipulation_env` environments and the task wrappers around it in `mobileRL.env.tasks` and `mobileRL.env.tasks_chained`. 
Originally published in [[1]](#kinematic-feasibility), demonstrations can be found on [Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning](http://kinematic-rl.cs.uni-freiburg.de/).

The [MobileRLLearner](#src.control.mobile_manipulation.mobile_manipulation_learner.py) class has the following public methods:

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
- **restore_model_path**: *str, default=None*
  Path to load checkpoints from. Set to 'pretrained' to load one of the provided checkpoints.
- **temp_path**: *str, default='temp'*  
  Specifies a path where the algorithm stores log files and saves checkpoints.
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
The repository consists of two main parts: a training environment written in C++ and connected to Python through bindings and the RL agents written in Python 3.

As not all ROS packages work with python3, the setup relies on running the robot-specific packages in a python2 environment
and our package in a python3 environment.
The environment was tested for Ubuntu 18.04 and ROS melodic.

###### Installation
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings

    sudo apt install ros-[version]-pybind11-catkin

We provide implementations for the PR2, PAL Tiago and Toyota HSR robots.
The following outlines the installation for the PR2.
For the other robots please follow the official guides to install the respective ROS dependencies.
We recommend to use separate catkin workspaces for each.

Install the openDR dependencies and activate it's python environment

Install libgp: `https://github.com/mblum/libgp.git`

Create a catkin workspace (ideally a separate one for each robot)

    mkdir ~/catkin_ws
    cd catkin_ws

Copy or symlink openDR's mobile_manpipulation module into `./src`

    ln -s ln -s [opendr]/src/control/mobile_manipulation src/

Configure the workspace to use your environment's python3 (adjust path according to your executable)

    catkin config -DPYTHON_EXECUTABLE=/home/honerkam/miniconda3/envs/opendr/bin/python -DPYTHON_INCLUDE_DIR=/home/honerkam/miniconda3/envs/opendr/include/python3.7m -DPYTHON_LIBRARY=/home/honerkam/miniconda3/envs/opendr/lib/libpython3.7m.so

Build the workspace and source the setup file

    catkin build
    source devel/setup.bash

Tiago additionally requires small modifications to the robot descriptions to use the correct fixed joints. 
Replace the following files after installing the Tiago packages:

    cp [opendr]/src/control/mobile_manipulation/robots_world/tiago/modified_tiago.srdf.em src/tiago_moveit_config/config/srdf/tiago.srdf.em
    cp [opendr]/src/control/mobile_manipulation/robots_world/tiago/modified_tiago_pal-gripper.srdf src/tiago_moveit_config/config/srdf/tiago_pal-gripper.srdf
    cp [opendr]/src/control/mobile_manipulation/robots_world/tiago/modified_gripper.urdf.xacro src/pal_gripper/pal_gripper_description/urdf/gripper.urdf.xacro
    cp [opendr]/src/control/mobile_manipulation/robots_world/tiago/modified_wsg_gripper.urdf.xacro src/pal_wsg_gripper/pal_wsg_gripper_description/urdf/gripper.urdf.xacro

##### Run
1. start a roscore

        roscore

2a. training or evaluation in the analytical environment only:

        roslaunch mobile_manipulation_rl pr2_analytical.launch
   
2b. evaluation in gazebo: _instead_ of 2a start gazebo with the pr2 robot as well moveit. Please run from outside the conda environment with a python2 interpreter.

        roslaunch pr2_gazebo pr2_empty_world.launch
        roslaunch pr2_moveit_config move_group.launch

4. Run the demo script

        cd opendr_internal/projects/control/mobile_manipulation/
        export PYTHONPATH=/home/honerkam/repos/opendr_internal/src:=$PYTHONPATH
        python src/project_mobile_manipulation/mobile_manipulation_demo.py

5. [Visualisation] start rviz:

        rviz -d rviz_config.config

For HSR / Tiago: 
- Roslaunch commands can be found in `mobile_manipulation/mobileRL/handle_launchfiles.py`
- adjust the reference frame in rviz from `odom` to `odom combined`


#### Examples
* **Training in the analytical environment and evaluation in Gazebo on a Door Opening task**.
  As described above, install ROS and build the workspace. Then start a roscore and the launch files.
  ```python
    import rospy
    from pathlib import Path
    from control.mobile_manipulation.mobileRL.evaluation import evaluate_on_task
    from control.mobile_manipulation.mobileRL.utils import create_env
    from control.mobile_manipulation.mobile_manipulation_learner import MobileRLLearner

    # need a node to for visualisation
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"

    # create envs
    env_config = {
      'env': 'pr2',
      'penalty_scaling': 0.01,
      'time_step_world': 0.02,
      'seed': 42,
      'strategy': 'dirvel',
      'world_type': 'sim',
      'init_controllers': True,
      'perform_collision_check': True,
      'vis_env': True,
      'transition_noise_base': 0.015,
      'ik_fail_thresh': 20,
      'ik_fail_thresh_eval': 100,
      'learn_vel_norm': -1,
      'slow_down_real_exec': 2,
      'head_start': 0,
      'node_handle': 'train_env'
    }
  
    env = create_env(env_config, task='rndstartrndgoal', node_handle="train_env", wrap_in_dummy_vec=True, flatten_obs=True)
    eval_config = env_config.copy()
    eval_config["transition_noise_base"] = 0.0
    eval_config["ik_fail_thresh"] = env_config['ik_fail_thresh_eval']
    eval_config["node_handle"] = "eval_env"
    eval_env = create_env(eval_config, task=eval_config["task"], node_handle="eval_env", wrap_in_dummy_vec=True, flatten_obs=True)

    agent = MobileRLLearner(env,
                            checkpoint_after_iter=20_000,
                            temp_path=logpath,
                            device='cpu')

    # train on random goal reaching task
    agent.fit(env, val_env=eval_env)

    # evaluate on door opening in gazebo
    evaluate_on_task(env_config, eval_env_config=eval_config, agent=agent, task='door', world_type='gazebo')

    rospy.signal_shutdown("We are done")
  ```

* **Evaluate a pretrained model**.
  ```python
    import rospy
    from pathlib import Path
    from control.mobile_manipulation.mobileRL.evaluation import evaluate_on_task
    from control.mobile_manipulation.mobileRL.utils import create_env
    from control.mobile_manipulation.mobile_manipulation_learner import MobileRLLearner

    # need a node to for visualisation
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"

    # create envs
    eval_config = {
      'env': 'pr2',
      'penalty_scaling': 0.01,
      'time_step_world': 0.02,
      'seed': 42,
      'strategy': 'dirvel',
      'world_type': 'sim',
      'init_controllers': True,
      'perform_collision_check': True,
      'vis_env': True,
      'transition_noise_base': 0.0,
      'ik_fail_thresh': 100,
      'learn_vel_norm': -1,
      'slow_down_real_exec': 2,
      'head_start': 0,
      'node_handle': 'eval_env'
    }
  
    eval_env = create_env(eval_config, task=eval_config["task"], node_handle="eval_env", wrap_in_dummy_vec=True, flatten_obs=True)

    agent = MobileRLLearner(eval_env,
                            checkpoint_after_iter=0,
                            checkpoint_load_iter=1_000_000,
                            restore_model_path='pretrained',
                            temp_path=logpath,
                            device='cpu')

    # evaluate on door opening in gazebo
    evaluate_on_task(eval_config, eval_env_config=eval_config, agent=agent, task='door', world_type='gazebo')

    rospy.signal_shutdown("We are done")
  ```


#### Notes

##### HSR
The HSR environment relies on packages that are part of the proprietory HSR simulator. If you have an HSR account with Toyota,
please follow these steps to use the environment. Otherwise ignore this section to use the other environments we provide.

- Check the commented out parts in the `# HSR` section as well as the building of the workspace further below in the `Dockerfile` to install the requirements.
- Comment in the following lines in `CMakeLists.txt`:

      #  tmc_robot_kinematics_model      
      add_library(robot_hsr src/robot_hsr.cpp)
      target_link_libraries(robot_hsr worlds myutils ${catkin_LIBRARIES})

  and add them to `pybind_add_module()` and `target_link_libraries()` two lines below that.
- Comment in the hsr parts in `src/pybindings` and the import of HSREnv in `mobileRL/envs/robotenv.py` to create the python bindings
- Some HSR launchfiles are not opensource either and might need some small adjustments 

#### References
<a name="kinematic-feasibility" href="https://arxiv.org/abs/2101.05325">[1]</a> Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning,
[arXiv](https://arxiv.org/abs/2101.05325).
