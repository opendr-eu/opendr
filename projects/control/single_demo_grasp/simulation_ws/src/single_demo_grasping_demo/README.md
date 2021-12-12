# panda_webots

Simulation of the Franka Emika robotic arm in webots, controlled using ROS and moveit. 

## Installation 

Install webots, ROS, and moveit. 

## Usage

Start the simulation:
```
roslaunch panda_webots panda_sim.launch
```

Start the ROS interface: 
```
roslaunch panda_webots panda_controller.launch
```

Moveit control example:
```
roslaunch panda_webots panda_sim_control.launch
``` 
