# OpenDR EAGER

This folder contains the Engine Agnostic Gym Environment for Robotics (EAGER) toolkit. 
This toolkit allows users to create Gym environments that can be used with both real and simulated robots.

## Install

If you have not done this yet, first install the OpenDR toolkit dependencies.
Go to the root of the OpenDR toolkit and run:
```
source bin/setup.bash
```
Then run:
```
make install_runtime_dependencies
```
Now the EAGER toolkit can be installed (mind the source command):
```
source $OPENDR_HOME/src/opendr/control/gym_tools/eager/install_eager.sh
```
You can check if installation was succesful by running one of the examples:
```
roslaunch opendr_example example_action_processing.launch
```
or
```
roslaunch opendr_example example_multi_robot.launch
```
or
```
roslaunch opendr_example example_switch_engine.launch
```
Note that the toolkit can now be used from the terminal in which the install script is run.
For usage in another terminal either run the installation script there too (with source) or run:
```
source $OPENDR_HOME/lib/ur_ws/devel/setup.bash
source $OPENR_HOME/lib/eager_ws/devel/setup.bash
```
