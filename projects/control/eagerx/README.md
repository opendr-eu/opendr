[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# EAGERx

Engine Agnostic Gym Environment with Reactive extension (EAGERx) is a toolkit that will allow users to apply (deep) reinforcement learning for both simulated and real robots as well as combinations thereof.
The source code of EAGERx is available [here](https://github.com/eager-dev/eagerx)


## Installation

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

Follow the OpenDR [installation instructions](https://github.com/tasostefas/opendr_internal/wiki/Development-System-Setup/).

## Examples

After installation of the OpenDR toolkit, you should first source the EAGERx catkin_ws in order to run one of the available examples:

```bash
source $OPENDR_HOME/lib/catkin_ws_eagerx/devel/setup.bash
```

Next, you can run one of the examples after starting a ros core in a separate terminal:

```bash
roscore
```

Then, in a new terminal run:

```bash
rosrun eagerx_example_opendr [demo_name]
```

where possible values for [demo_name] are:
- demo_1_full_state
- demo_2_image
- demo_3_classifier
- demo_4_pid

## Citing EAGERx

To cite EAGERx in publications:
```bibtex
@misc{eagerx,
  author = {Van der Heijden, Bas and Luijkx, Jelle},
  title = {EAGERx: Engine Agnostic Gym Environment with Reactive extension},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eager-dev/eagerx}},
}
```
