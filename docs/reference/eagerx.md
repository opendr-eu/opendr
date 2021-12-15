## EAGERx project

Engine Agnostic Gym Environment with Reactive extension (EAGERx) is a toolkit within OpenDR that allows users to apply (deep) reinforcement learning for both simulated and real robots as well as combinations thereof.
The toolkit serves as bridge between the popular reinforcement learning toolkit [OpenAI Gym](https://gym.openai.com/) and robots that can either be real or simulated.
The EAGERx toolkit makes use of the widely used ROS framework for communication and ReactiveX synchronization of actions and observations.
Nonetheless, thanks to the flexible design of the toolkit, it is possible for users to create customized bridges for robots without ROS support.

The source code for EAGERx is available [here](https://github.com/eager-dev/eagerx).

### Key Functionalities and Features


| **Functionality/Feature**                                           | **EAGERx**         |
| ------------------------------------------------------------------- | -------------------|
| User-friendly creation of Gym environments for robot control tasks  | :heavy_check_mark: |
| Synchronization of actions and observations in simulators           | :heavy_check_mark: |
| Processing of data streams                                          | :heavy_check_mark: |
| Switching between physics engines                                   | :heavy_check_mark: |

Documentation is available online: [https://eagerx.readthedocs.io](https://eagerx.readthedocs.io)


### Examples

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

After installation of the OpenDR toolkit, you can run one of the available examples as follows.

First source the workspace:

```bash
source $OPENDR_HOME/projects/control/eagerx/eagerx_ws/devel/setup.bash
```

and start ROS core:

```bash
roscore
```

Then in a separate terminal, after sourcing again, you can run one of the demos:

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
