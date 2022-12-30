# mobile_manipulation module

The *mobile_manipulation* module contains the *MobileRLLearner* class, which inherits from the abstract class *LearnerRL*.

### Class MobileRLLearner
Bases: `engine.learners.LearnerRL`

The *MobileRLLearner* class is an RL agent that can be used to train wheeled robots for mobile manipulation in conjunction with the
`mobileRL.env.mobile_manipulation_env` environments and the task wrappers around it in `mobileRL.env.tasks` and `mobileRL.env.tasks_chained`.
Originally published in [[1]](#kinematic-feasibility), demonstrations can be found on [Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning](http://kinematic-rl.cs.uni-freiburg.de/).

The [MobileRLLearner](/src/opendr/control/mobile_manipulation/mobile_manipulation_learner.py) class has the following public methods:

#### `MobileRLLearner` constructor

Constructor parameters:

- **env**: *gym.Env*\
  Reinforcment learning environment to train or evaluate the agent on.
- **lr**: *float, default=1e-5*\
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=1_000_000*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*\
  Specifies the batch size during training.
- **lr_schedule**: *{'', 'linear'}, default='linear'*\
  Specifies the learning rate scheduler to use. Empty to use a constant rate.
- **lr_end**: *float, default=1e-6*\
  Specifies the final learning rate if a lr_schedule is used.
- **backbone**: *{'MlpPolicy'}, default='MlpPolicy'*\
  Specifies the architecture for the RL agent.
- **checkpoint_after_iter**: *int, default=20_000*\
  Specifies per how many training steps a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **restore_model_path**: *str, default=None*\
  Path to load checkpoints from. Set to 'pretrained' to load one of the provided checkpoints.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm stores log files and saves checkpoints.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **seed**: *int, default=None*\
  Random seed for the agent. If None a random seed will be used.
- **buffer_size**: *int, default=100_000*\
  Size of the replay buffer.
- **learning_starts**: *int, default=0*\
  Number of environment steps with a random policy before starting training.
- **tau**: *float, default=0.001*\
  Polyak averaging of the target network.
- **gamma**: *float, default=0.99*\
  Discount factor.
- **explore_noise**: *float, default=0.5*\
  Strength of the exploration noise.
- **explore_noise_type**: *{'normal', 'OU'}, default=normal*\
  Type of exploration noise, either normal or Ornstein Uhlenbeck (OU) noise.
- **ent_coef**: *{'auto', int}, default='auto'*\
  Entropy coefficient for SAC, 'auto' to learn the coefficient.
- **nr_evaluations**: *int, default=50*\
  Number of episodes to evaluate over.
- **evaluation_frequency**: *int, default=20_000*\
  Number of steps after which to episodically evaluate during training.

#### `MobileRLLearner.fit`
```python
MobileRLLearner.fit(self, env, val_env, logging_path, silent, verbose)
```

Train the agent on the environment.

Parameters:

- **env**: *gym.Env, default=None*\
  If specified use this env to train.
- **val_env**: *gym.Env, default=None*\
  If specified periodically evaluate on this environment.
- **logging_path**: *str, default=''*\
  Path for logging and checkpointing.
- **silent**: *bool, default=False*\
  Disable verbosity.
- **verbose**: *bool, default=True*\
  Enable verbosity.


#### `MobileRLLearner.eval`
```python
MobileRLLearner.eval(self, env, name_prefix='', nr_evaluations: int = None)
```
Evaluate the agent on the specified environment.

Parameters:

- **env**: *gym.Env, default=None*\
  Environment to evaluate on.
- **name_prefix**: *str, default=''*\
  Name prefix for all logged variables.
- **nr_evaluations**: *int, default=None*\
  Number of episodes to evaluate over.


#### `MobileRLLearner.save`
```python
MobileRLLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `MobileRLLearner.load`
```python
MobileRLLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.



#### ROS Setup
The repository consists of two main parts: a training environment written in C++ and connected to Python through bindings and the RL agents written in Python 3.

This means that the training environment relies on a running moveit node for initialisation.
The dependencies for this module automatically set up and compile a catkin workspace with all required modules.
To start required ROS nodes, please run the following before using the `MobileRLLearner` class:

```sh
source ${OPENDR_HOME}/projects/python/control/mobile_manipulation/mobile_manipulation_ws/devel/setup.bash
roslaunch mobile_manipulation_rl [pr2,tiago]_analytical.launch
````

The environment requires a working ROS installation. The makefile will install the packages according to the ROS_DISTRO environment variable.

##### Visualisation
All visualisations are done through rviz. For this start rviz with the provided configuration file as follows.
To visualise the TIAGo robot additionally adjust the reference frame in rviz from `odom` to `odom combined`.
```sh
rviz -d rviz_config.config
```


#### Examples
* **Training in the analytical environment and evaluation in Gazebo on a Door Opening task**.
As described above, install ROS and build the workspace.
Then source the catkin workspace and run the launch file as described in the `ROS Setup` section above.
  ```python
    import rospy
    from pathlib import Path
    from opendr.control.mobile_manipulation import evaluate_on_task
    from opendr.control.mobile_manipulation import create_env
    from opendr.control.mobile_manipulation import MobileRLLearner

    # need a node for visualisation
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"

    # create envs
    env_config = {
      'env': 'pr2',
      'penalty_scaling': 0.01,
      'time_step': 0.02,
      'seed': 42,
      'strategy': 'dirvel',
      'world_type': 'sim',
      # set this to true to evaluate in gazebo
      'init_controllers': False,
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
    eval_env = create_env(eval_config, task="rndstartrndgoal", node_handle="eval_env", wrap_in_dummy_vec=True, flatten_obs=True)

    agent = MobileRLLearner(env,
                            checkpoint_after_iter=20_000,
                            temp_path=logpath,
                            device='cpu')

    # train on random goal reaching task
    agent.fit(env, val_env=eval_env)

    # evaluate on door opening in analytical environement
    evaluate_on_task(env_config, eval_env_config=eval_config, agent=agent, task='door', world_type='sim')

    rospy.signal_shutdown("We are done")
  ```

* **Evaluate a pretrained model**.
  Source the catkin workspace and run the launch file as described in the `ROS Setup` section above. Then run
  ```python
    import rospy
    from pathlib import Path
    from opendr.control.mobile_manipulation import evaluate_on_task
    from opendr.control.mobile_manipulation import create_env
    from opendr.control.mobile_manipulation import MobileRLLearner

    # need a node for visualisation
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"

    # create envs
    eval_config = {
      'env': 'pr2',
      'penalty_scaling': 0.01,
      'time_step': 0.02,
      'seed': 42,
      'strategy': 'dirvel',
      'world_type': 'sim',
      # set this to true to evaluate in gazebo
      'init_controllers': False,
      'perform_collision_check': True,
      'vis_env': True,
      'transition_noise_base': 0.0,
      'ik_fail_thresh': 100,
      'learn_vel_norm': -1,
      'slow_down_real_exec': 2,
      'head_start': 0,
      'node_handle': 'eval_env',
      'nr_evaluations': 50,
    }

    eval_env = create_env(eval_config, task='rndstartrndgoal', node_handle="eval_env", wrap_in_dummy_vec=True, flatten_obs=True)

    agent = MobileRLLearner(eval_env,
                            checkpoint_after_iter=0,
                            checkpoint_load_iter=1_000_000,
                            restore_model_path='pretrained',
                            temp_path=logpath,
                            device='cpu')

    # evaluate on door opening in the analytical environment
    evaluate_on_task(eval_config, eval_env_config=eval_config, agent=agent, task='door', world_type='sim')

    rospy.signal_shutdown("We are done")
  ```
* **Execution in different environments**:\
  The trained agent and environment can also be directly executed in the real world or the gazebo simulator. For this first start the appropriate ros nodes for your robot. Then pass `world_type='world'` for real world execution or `world_type='gazebo'` for gazebo to the `evaluate_on_task()` function.


#### Performance Evaluation

Note that test time inference can be directly performed on a standard CPU.
As this achieves very high control frequencies, we do not expect any benefits through the use of accelerators (GPUs).

TABLE-1: Control frequency in Hertz.

| Model    | AMD Ryzen 9 5900X (Hz) |
| -------- | ---------------------- |
| MobileRL | 2200                   |


TABLE-2: Success rates in percent.

| Model    | GoalReaching | Pick&Place | Door Opening | Drawer Opening |
| -------- | ------------ | ---------- | ------------ | -------------- |
| PR2      | 90.2%        | 97.0%      | 94.2%        | 95.4%          |
| Tiago    | 71.6%        | 91.4%      | 95.3%        | 94.9%          |
| HSR      | 75.2%        | 93.4%      | 91.2%        | 90.6%          |


TABLE-3: Platform compatibility evaluation.

| Platform                                     | Test results  |
| -------------------------------------------- | ------------- |
| x86 - Ubuntu 20.04 (bare installation - CPU) | Pass          |
| x86 - Ubuntu 20.04 (bare installation - GPU) | Pass          |
| x86 - Ubuntu 20.04 (pip installation)        | Not supported |
| x86 - Ubuntu 20.04 (CPU docker)              | Pass          |
| x86 - Ubuntu 20.04 (GPU docker)              | Pass          |
| NVIDIA Jetson TX2                            | Not tested    |
| NVIDIA Jetson Xavier AGX                     | Not tested    |


#### Notes

##### HSR
The HSR environment relies on packages that are part of the proprietary HSR simulator.
If you have an HSR account with Toyota, please follow these steps to use the environment.
Otherwise ignore this section to use the other environments we provide.

- Check the commented out parts in the `# HSR` section as well as the building of the workspace further below in the `Dockerfile` to install the requirements.
- Comment in the following lines in `CMakeLists.txt`:

      #  tmc_robot_kinematics_model
      add_library(robot_hsr src/robot_hsr.cpp)
      target_link_libraries(robot_hsr worlds myutils ${catkin_LIBRARIES})

  and add them to `pybind_add_module()` and `target_link_libraries()` two lines below that.
- Comment in the hsr parts in `src/pybindings` and the import of HSREnv in `mobileRL/envs/robotenv.py` to create the python bindings
- Some HSR launchfiles are not open source either and might need some small adjustments

#### References
<a name="kinematic-feasibility" href="https://arxiv.org/abs/2101.05325">[1]</a> Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning,
[arXiv](https://arxiv.org/abs/2101.05325).
