# OpenDR End-to-end Local Planning

This folder contains the OpenDR Learner class for end-to-end planning tasks. 
This method uses reinforcement learning to train an agent that is able to plan 

## Simulation RL environment setup

The environment includes an Ardupilot controlled quadrotor in Webots simulation. 
Installation of Ardupilot software follows this link: `https://github.com/ArduPilot/ardupilot`

The files under `end_to_end_planning/ardupilot` should be replaced under the installation of Ardupilot.
In order to run the Ardupilot in Webots 2021a controller codes should be replaced. 
For older versions of Webots, these files can be skipped.
The world file for the environment is provided 
under `/ardupilot/libraries/SITL/examples/webots/worlds/Agri2021a.wbt`.

Install `mavros` package into catkin workspace for ROS communication with Ardupilot. 

## Running the environment

The following steps should be executed to have a ROS communication between Gym environment and simulation.
- Start the Webots and run the world file.
- Run following script from Ardupilot directory: `./libraries/SITL/examples/Webots/dronePlus.sh`
- `roslaunch mavros apm.launch`
- Run `children_robot` node
- Run `take_off` node
- Run `range_image` node

After these steps the gym environment can send action comments to the 
simulated drone and receive depth image and pose information from simulation. 

