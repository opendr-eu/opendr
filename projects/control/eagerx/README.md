[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# EAGERx

Engine Agnostic Gym Environment with Reactive extension (EAGERx) is a toolkit that will allow users to apply (deep) reinforcement learning for both simulated and real robots as well as combinations thereof.
Documentation is available [here](../../../docs/reference/eagerx.md).
The source code of EAGERx is available [here](https://github.com/eager-dev/eagerx)

### Installation

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

Follow the OpenDR installation instructions.
Next, one should also install the appropriate runtime dependencies:

```bash
cd $OPENDR_HOME
make install_runtime_dependencies
```

Now the user is ready to go!


### Examples

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

After installation of the OpenDR toolkit, you can run one of the available examples as follows.

First source the workspace:

```bash
source $OPENDR_HOME/projects/control/eagerx/eagerx_ws/devel/setup.bash
```

Now you can run one of the demos in the terminal where you sourced the workspace:

```bash
source $OPENDR_HOME/projects/control/eagerx/eagerx_ws/devel/setup.bash
rosrun eagerx_example_opendr [demo_name]
```

where possible values for [demo_name] are:
- **demo_1_full_state**: Here, we wrap the OpenAI gym within EAGERx.
The agent learns to map low-dimensional angular observations to torques.
- **demo_2_pid**: Here, we add a PID controller, tuned to stabilize the pendulum in the upright position, as a pre-processing node.
The agent now maps low-dimensional angular observations to reference torques.
In turn, the reference torques are converted to torques by the PID controller, and applied to the system.
- **demo_3_classifier**: Instead of using low-dimensional angular observations, the environment now produces pixel images of the pendulum.
In order to speed-up learning, we use a pre-trained classifier to convert these pixel images to estimated angular observations.
Then, the agent uses these estimated angular observations similarly as in 'demo_2_pid' to successfully swing-up the pendulum.


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
