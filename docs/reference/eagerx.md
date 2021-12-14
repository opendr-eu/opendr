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

After installation of the OpenDR toolkit, you can run one of the available examples as follows.
First, you need to start a ROS core in your terminal:

```bash
roscore
```

Then, in a separate terminal you should source the EAGERx catkin workspace:

```bash
source $OPENDR_HOME/projects/control/eagerx/eagerx_ws/devel/setup.bash
```

Now you can run one of the demos:

```bash
rosrun eagerx_example_opendr [demo_name]
```

where possible values for [demo_name] are:
- demo_1_full_state
- demo_2_image
- demo_3_classifier
- demo_4_pid
