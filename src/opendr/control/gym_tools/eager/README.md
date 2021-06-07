# OpenDR EAGER

This folder contains the Engine Agnostic Gym Environment for Robotics (EAGER) toolkit. This toolkit allows user to create Gym environments that can be used with both real and simulated robots.

## Install

If you have not done this yet, first install the OpenDR toolkit dependencies.
Go to the root of the OpenDR toolkit and run

```
source bin/setup.bash
```
Then run:
```
make install_runtime_dependencies
```
Now the EAGER toolkit can be installed:
```
source $OPENDR_HOME/src/opendr/control/gym_tools/eager/install_eager.sh
```
You can check if installation was succesful by running:
```
roslaunch opendr_example example_multi_robot.launch
```


