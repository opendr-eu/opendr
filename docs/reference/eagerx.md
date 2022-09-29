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


### EAGERx Demos

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

1. **[demo_full_state](../../projects/python/control/eagerx/demos/demo_full_state.py)**:  
   Here, we wrap the OpenAI gym within EAGERx.
   The agent learns to map low-dimensional angular observations to torques.
2. **[demo_pid](../../projects/python/control/eagerx/demos/demo_pid.py)**:   
   Here, we add a PID controller, tuned to stabilize the pendulum in the upright position, as a pre-processing node.
   The agent now maps low-dimensional angular observations to reference torques.
   In turn, the reference torques are converted to torques by the PID controller, and applied to the system.
3. **[demo_classifier](../../projects/python/control/eagerx/demos/demo_classifier.py)**:   
   Instead of using low-dimensional angular observations, the environment now produces pixel images of the pendulum.
   In order to speed-up learning, we use a pre-trained classifier to convert these pixel images to estimated angular observations.
   Then, the agent uses these estimated angular observations similarly as in 'demo_2_pid' to successfully swing-up the pendulum.

Example usage:
```bash
cd $OPENDR_HOME/projects/python/control/eagerx/demos
python3 [demo_name]
```

where possible values for [demo_name] are: *demo_full_state.py*, *demo_pid.py*, *demo_classifier.py*

Setting `--device cpu` performs training and inference on CPU.  
Setting `--name example` sets the name of the environment.  
Setting `--eps 200` sets the number of training episodes.  
Setting `--eval-eps 10` sets the number of evaluation episodes.
Adding `--render` enables rendering of the environment.

### Performance Evaluation

In this subsection, we attempt to quantify the computational overhead that the communication protocol of EAGERx introduces.
Ultimately, an EAGERx environment consists of nodes (e.g. sensors, actuators, classifiers, controllers, etc…) that communicate with each other via the EAGERx’s reactive communication protocol to ensure I/O synchronisation.
We create an experimental setup where we interconnect a set of the same  nodes in series and let every node run at the same simulated rate (1 Hz).
The nodes perform no significant computation in their callback, and use small messages (ROS message: std.msgs.msg/UInt64) in terms of size (Mb).
Hence, the rate at which this environment can be simulated is mostly determined by the computational overhead of the protocol and the hardware used during the experiment (8 core - Intel Core i9-10980HK Processor).
We will record the real-time rate (Hz) at which we are able to simulate the environment for a varying number of interconnected nodes, synchronisation modes (sync vs. async), and concurrency mode (multi-threaded vs multi-process).
In async mode, every node will produce outputs at the set simulated rate (1 Hz) times a provided real-time factor (i.e. real-time rate = real-time factor * simulated rate).
This real-time factor is set experimentally at the highest value, while each node can still keep its simulated rate.
In sync mode, nodes are connected reactively, meaning that every node will wait for an input from the preceding node, before producing an output message to the node that succeeds it.
This means that we do not need to set a real-time factor.
Instead, nodes will run as fast as possible, while adhering to this simple rule.
The recorded rate provides an indication of the computational overhead that the communication protocol of EAGERx introduces.
The results are presented in table below.

|            | Multi-threaded |            | Multi-process |            |
|------------|:--------------:|:----------:|:-------------:|:----------:|
| # of nodes |    Sync (Hz)   | Async (Hz) |   Sync (Hz)   | Async (Hz) |
| 4          |       800      |     458    |      700      |    1800    |
| 5          |       668      |     390    |      596      |    1772    |
| 6          |       576      |     341    |      501      |    1770    |
| 7          |       535      |     307    |      450      |    1691    |
| 12         |       354      |     200    |      279      |    1290    |

The platform compatibility evaluation is also reported below:

| Platform                                     | Test results |
|----------------------------------------------|:------------:|
| x86 - Ubuntu 20.04 (bare installation - CPU) |     Pass     |
| x86 - Ubuntu 20.04 (bare installation - GPU) |     Pass     |
| x86 - Ubuntu 20.04 (pip installation)        |     Pass     |
| x86 - Ubuntu 20.04 (CPU docker)              |     Pass     |
| x86 - Ubuntu 20.04 (GPU docker)              |     Pass     |
| NVIDIA Jetson TX2                            |     Pass     |
