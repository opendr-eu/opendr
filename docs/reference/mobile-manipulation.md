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
- **lr**: *float, default=4e-5*  
  Specifies the initial learning rate to be used during training.
- **epochs**: *int, default=280*  
  Specifies the number of epochs the training should run for.
- **batch_size**: *int, default=80*  
  Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **backbone**: *{'mobilenet, 'mobilenetv2', 'shufflenet'}, default='mobilenet'*  
  Specifies the backbone architecture.
- **lr_schedule**: *str, default=' '*  
  Specifies the learning rate scheduler. Please provide a function that expects to receive as a sole argument the used optimizer.
- **temp_path**: *str, default='temp'*  
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files. Moreover the JSON file that contains the evaluation detections is saved here.
- **checkpoint_after_iter**: *int, default=5000*  
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.



#### References
<a name="kinematic-feasibility" href="https://arxiv.org/abs/2101.05325">[1]</a> Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning,
[arXiv](https://arxiv.org/abs/2101.05325).



## WIP

Repository providing the source code for the paper "Learning kinematic feasibility through reinforcement learning", see the [project website](https://rl.uni-freiburg.de/research/kinematic-feasibility-rl).  
Please cite the paper as follows:

    @article{honerkamp2021learning,
      title={Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning}, 
      author={Daniel Honerkamp and Tim Welschehold and Abhinav Valada},
      journal={arXiv preprint arXiv:2101.05325},
      year={2021},
    }
    
## Docker
Easiest way to get started is to pull the docker image 
	
	*TODO* docker pull dhonerkamp/modulation_rl:latest

The current implementation relies on the Weights And Biases library for logging.
So create a python3 environment with wandb (free account required) and run `wandb login` to login to your account.
Alternatively first run `wandb disabled` in your shell to run it without an account and without logging any results 
(evaluations still get printed to stdout).

The following commands also assume that you have docker installed, and to use a CUDA-capable GPUs also the nvidia-docker driver. To only use the CPU remove the `--gpus` flag. 

To train a model simply run the following command. Currently we provide environments for the PR2, Tiago and HSR (for the HSR please read the note below)

    env=pr2  # all lower-case
    wandb docker-run --rm --gpus=all -e ROBOT=$env -e STARTUP=$env dhonerkamp/modulation_rl:latest bash ros_startup_incl_train.sh "python scripts/main.py --load_best_defaults --env $env" 

We provide trained model checkpoints for each of the robots in `scipts/model_checkpoints`. To evaluate them add the following flags

    --start_launchfiles_no_controllers=no --vis_env --evaluation_only --restore_model

To evaluate a previous run after you have logged it with wandb replace `--restore_model` with `--resume_id=${wandb_id}`.
(The `--start_launchfiles_no_controllers=yes` flag allows to train faster by not running Gazebo in the background, but is required for evaluations in Gazebo, 
`--vis_env` publishes all rviz markers). 

## HSR
The HSR environment relies on packages that are part of the proprietory HSR simulator. If you have an HSR account with Toyota, 
please follow these steps to use the environment. Otherwise ignore this section to use the other environments we provide.

- Check the commented out parts in the `# HSR` section as well as the building of the workspace further below in the `Dockerfile` to install the requirements.
- Comment in the following lines in `CMakeLists.txt`:
    
      add_library(dynamic_system_hsr src/dynamic_system_hsr.cpp)
      target_link_libraries(dynamic_system_hsr modulation modulation_ellipses utils ${catkin_LIBRARIES})

  and add them to `pybind_add_module()` and `target_link_libraries()` two lines below that.
- Comment in the hsr parts in `src/pybindings` and the import of HSREnv in `scripts/modulation/envs/modulationEnv.py` to create the python bindings
- Now either build the Dockerfile or rebuild a local workspace (we recommend to use separate catkin workspaces for each robot)


## Evaluating on your own task
Tasks are implemented as environment wrappers in `scripts/modulation/envs/tasks.py`.
To generate end-effector motions we provide both the `linearPlanner` using a dynamic linear system and the `OLD_GMMPlanner` 
using an imitation learning approach used in the paper. Both are implemented in C++.

To construct a new task add a new wrapper to `tasks.py`, add it's name to the `--task` flag 
in `scripts/modulation/utils.py` and define how to construct it in `scripts/modulation/envs/env_utils.py`.

To use different end-effector motions check the planners mentioned above.

## Local installation
For development or qualitative inspection of the behaviours in rviz or gazebo it can be easier to install the setup locally.
The following illustrates the main steps to do this for the PR2. 
As the different robots come with different ROS dependencies, please use the `Dockerfile` as the full guide to install them.

The repository consists of two main parts: a training environment written in C++ and connected to python through bindings and the RL agents written in python3.

As not all ROS packages work with python3, the setup relies on running the robot-specific packages in a python2 environment
and our package in a python3 environment.
The environment was tested for Ubunto 18.04 and ROS melodic.

### Install
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings
        
    sudo apt install ros-[version]-pybind11-catkin
        
Install moveit and the pr2
    
    sudo apt-get install ros-[version]-moveit
    sudo apt-get install ros-[version]-pr2-simulator
    sudo apt-get install ros-[version]-moveit-pr2
   
If interested in the geometric baseline, also install libgp: `https://github.com/mblum/libgp.git`

Create a catkin workspace (ideally a separate one for each robot)

    mkdir ~/catkin_ws
    cd catkin_ws

Fork the repo and clone into `./src`
    
    cd src
    git clone [url] src/modulation_rl

Create a python environment. We recommend using conda, which requires to first install Anaconda or Miniconda. Then do

    conda env create -f src/modulation_rl/environment.yml
    conda activate modulation_rl

Configure the workspace it to use your environment's python3 (adjust path according to your version)

    catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so
    
Build the workspace
    
    catkin build
    
Each new build of the ROS / C++package requires a
    
    source devel/setup.bash
    
To be able to visualise install rviz

    http://wiki.ros.org/rviz/UserGuide
    
For more details and how to install the ROS requirements for the other robots please check the `Dockerfile`. It also contains further details to packages you might need to build from source.


### Run
1. start a roscore
        
        roscore
2. start gazebo with the pr2 robot
        
        roslaunch pr2_gazebo pr2_empty_world.launch
3. start moveit

        roslaunch pr2_moveit_config move_group.launch
4. Run the main script

        python src/modulation_rl/scripts/main.py
5. [Only to visualise] start rviz:

        rviz -d src/modulation_rl/rviz_config[_tiago_hsr].config
        

## Troubleshooting
- Library conflicts: error message either around `cv2` or `libgcc_s.so.1 must be installed for pthread_cancel to work`:
    Solution: rename cv2 installed by ROS: 
    https://stackoverflow.com/questions/48039563/import-error-ros-python3-opencv
