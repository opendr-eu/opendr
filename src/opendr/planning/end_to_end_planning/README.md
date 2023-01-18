# End-to-end Planning

This folder contains the OpenDR Learner class for end-to-end planning tasks. 
This method uses reinforcement learning to train an agent that is able to generate local motion plans for a quadrotor UAV equipped with a depth camera. 

### Using non-dynamic simulation environment

The end-to-end planning agent is interacting with gym environment which communicates with Webots.
The environment is provided with the [webots world](../../../../src/opendr/planning/end_to_end_planning/envs/webots/worlds/train-no-dynamic-random-obstacles.wbt)
that needs to be opened with Webots version 2022b in order to demonstrate the end-to-end planner.

### Using Ardupilot simulation environment

The environment includes an optional Ardupilot controlled quadrotor in Webots for dynamic simulation.
The environment required to be initiated with argument `no_dynamics=False` 
For the installation of Ardupilot instructions are available [here](https://github.com/ArduPilot/ardupilot).

The required files to complete Ardupilot setup can be downloaded by running [download_ardupilot_files.py](../../../../src/opendr/planning/end_to_end_planning/download_ardupilot_files.py) script.
The downloaded files (zipped as `ardupilot.zip`) should be replaced under the installation of Ardupilot.
In order to run Ardupilot in Webots 2021a, controller codes should be replaced. (For older versions of Webots, these files can be skipped.)
The world file for the environment is provided under `/ardupilot/libraries/SITL/examples/webots/worlds/` for training and testing.

Install `mavros` package for ROS communication with Ardupilot.
Instructions are available [here](https://github.com/mavlink/mavros/blob/master/mavros/README.md#installation).
Source installation is recommended.

### Running Ardupilot environment

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
  
After these steps the [UAVDepthPlanningEnv](../../../../src/opendr/planning/end_to_end_planning/envs/UAV_depth_planning_env.py) gym environment can send action comments to the simulated drone and receive depth image and pose information from simulation.
