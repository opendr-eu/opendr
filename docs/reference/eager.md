## EAGER Toolkit

Engine Agnostic Gym Environment for Robotics (EAGER) is a toolkit that
allows users to apply (deep) reinforcement learning for both simulated
and real robots as well as combinations thereof. The toolkit serves as
bridge between the popular reinforcement learning toolkit OpenAI
Gym and robots that can either be real or
simulated. The EAGER toolkit makes use of the widely used ROS
framework for communication. Nonetheless, thanks to
the flexible design of the toolkit, it is possible for users to create a
customized bridge for robots without ROS support.

OpenAI Gym is a toolkit for evaluating reinforcement learning algorithms
in so-called Gym environments and is used for benchmarking (deep)
reinforcement learning algorithms by the scientific
community. One of the benefits of OpenAI Gym is
that users can easily evaluate state-of-the-art (deep) reinforcement
learning algorithms for their Gym environments, thanks to the
availability of algorithm implementations, such as Stable
Baselines. OpenAI Gym comes with a number of
environments, including simulated robots, classic control tasks and
Atari games.

For reinforcement learning in robotics it is often highly desirable to
train both in simulation and reality, because simulations allow to train
faster than real-time and are safer than training with real robots. At
the same time, real-world experience is required in many cases, because
model inaccuracies of the simulator are exploited by reinforcement
learning algorithms. However, creating Gym
environments for both real and simulated robots is currently a difficult
and time-consuming task, because it is challenging to synchronize
actions and observations, to communicate with both real and simulated
robots and to add or interchange objects in environments. Therefore, we
introduce EAGER, which allows users to create engine agnostic Gym
environments that can be used with different simulators, physics engines
and with real robots. Also, by choosing an approach that is based on
composition --- instead of inheritance --- adding robots, sensors and
other objects to an environment is reduced to a one-liner of code. The
key functionalities that EAGER will provide are:

1.  User-friendly creation and modification of Gym environments for
    robot control tasks.

2.  Integration with popular robot simulators
    Webots, PyBullet and
    Gazebo.

3.  Synchronization of actions and observations.

4.  The possibility to switch between and/or combine simulated and real
    robots.

5.  The possibility to add preprocessing of actions and observations.

6.  The possibility to add procedures for resetting the environment
    after an episode.

### Methodology

In this section we will describe how the aforementioned functionalities
will be provided.

\(1\) EAGER allows users to easily create and modify Gym environments
for robot control tasks. The toolkit is designed such that new
environments can be created without the need to redefine objects, such
as robots, sensors and actuators. In this way, adding an object to an
existing environment is as easy as adding the name of the object to the
list of environment objects. Naturally, the environment objects need to
be defined at least once. We will provide such a definition for the UR5e
robot and documentation on how to create new object definitions. We aim
to stimulate the robotics community to add and share robot definitions
to eventually have an extensive number of robots and sensors supported
by the EAGER toolkit. Moreover, since ROS-based code is present in the
backend of the toolkit, the toolkit can also be used by users without
ROS experience. Nevertheless, the toolkit provides enough flexibility
for experienced ROS users to exploit the possibilities of ROS. Namely,
EAGER will provide the possibility to create custom action and
observation processing nodes, implement reset procedures or even to
create a custom bridge for an alternative simulator or robot.

\(2\) EAGER will provide integration with three simulators that are
frequently used by the robotics and reinforcement learning communities,
i.e. PyBullet, Gazebo and Webots. Also, users can train with different
simulators in parallel and a base class is provided for creating a
"bridge" for other simulators or physics engines.

\(3\) Synchronization of actions and observations is vital for effective
reinforcement learning, because policy updates are based on state-action
pairs and corresponding rewards in reinforcement learning. In order to
pair actions and observations correctly, actions and observations are
obtained through ROS services, rather than ROS topics. In order to
ensure that actions and observations are returned when these services
are called, we make use of buffers with a queue size of one.

\(4\) Thanks to the engine agnostic property of EAGER environments,
users can easily switch between real and simulated robots. Also, real
and simulated robots could be trained in parallel, e.g., for simulator
tuning.

\(5\) A base class is provided for adding processing steps to actions. A
similar base class will be provided for observations in the future.
Processing of actions and observations can be useful, e.g., for checking
whether actions are collision free, for obtaining the end-effector's
position from joint states or for obtaining locations of objects that
are detected in RGB images. Thanks to the implementation of the base
classes, users only have to implement the processing step and do not
have to worry about communication and synchronization issues.

\(6\) A base class will be provided for reset procedures that will be
executed at the end of episodes. This will allow users to automate their
training procedures. This will also be provided for real-world training,
but in that case there are limitations to the controllability of states.

### Install

If you have not done this yet, first activate the OpenDR virtual environment, by going to the root of the OpenDR toolkit and running:
```
source bin/setup.bash
```
Now the EAGER toolkit can be installed (mind the source command):
```
source $OPENDR_HOME/projects/control/eager/install_eager.sh
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
source $OPENDR_HOME/projects/opendr_ws/devel/setup.bash
```

#### Troubleshooting

Currently, the EAGER toolkit has dependency conflicts with other tools of the OpenDR toolkit.
Therefore launching *example_switch_engine.launch* results in an error due to an OpenCV conflict.

### OpenDR Implementation


In this section the implementation details of EAGER in the OpenDR
toolkit are discussed. We will also make clear what parts of the EAGER
toolkit have been implemented for the first version, and what
functionality is planned as future work. Given an environment called *MyEnv*
that inherited from the *BaseEagerEnv* base class, the toolkit works as
follows.

*MyEnv* is engine agnostic, meaning that any engine (e.g. Pybullet,
Webots, real world) can be used by specifying it as an argument to
MyEnv's constructor. Within MyEnv's constructor, the *PhysicsBridge*,
related to the specified engine, is launched. We will refer to this
specific *PhysicsBridge* as the *EngineBridge*. *MyEnv* provides a
common interface in the form of a set of ROS services to the
*EngineBridge*. *MyEnv* expects the launched *EngineBridge* to either
initialize the requested services or subscribe to services advertised by
*MyEnv* that comprise this common interface. The different services,
corresponding to the various functions of *MyEnv*, is illustrated in
Figure.

When launched, the *EngineBridge* initializes an empty world. To
populate this world with an *Object*, MyEnv sends the *Object*'s ROS
packagename, unique namespace, and initialization arguments (e.g. base
location) to the *EngineBridge* via a ROS service that is called upon by
$\_init\_nodes(...)$. This allows the EngineBridge to look up the
*Object*'s config file (*.yaml*) that contains all the details in order
to spawn the object in the simulated world. EAGER views any sensor,
robot, or other physical entity as an *Object*. Each *Object* has a
dedicated ROS package that contains all details for it to be spawned in
any of the considered engines. Specifically, an *Object* is defined in
the aforementioned config file (.yaml) in the *Object*'s ROS package as
a collection of sensors, actuators, and states together with other
required details such as the location to the robot description (.urdf).

EAGER allows actions and observations to be processed by adding custom
processing nodes. For
example, if a custom process is added to a specific actuator of an
*Object*, the *EngineBridge* is setup to call the action service of that
custom process. In turn, the custom process calls the *Object*'s action
service such that it can process the action and send the processed
action to the *EngineBridge*. Sensor processing will be implemented in a
similar fashion. As of now a real reset node that uses actions to reset
an object to a desired state, has not been implemented.

#### Package structure

The EAGER toolkit within OpenDR is split into separate ROS packages.
This allows users to only include the objects, bridges en processing
nodes they intend to use.

-   *eager_core*: This package contains the base classes for objects,
    physics bridges, processor nodes, environments, and parameters,
    together with standard utility functions. It also contains the
    definition for the various ROS services and messages that comprise
    the interface between the EAGER environment and the physics bridge.
    This package is the backbone of the EAGER toolkit and is always
    required.

-   *eager_bridge\_\[name_engine\]*: This package concerns the physics
    bridge implementation for a specific engine (e.g. Webots, Pybullet,
    reality).

-   *eager_sensor\_\[name_sensor\]*: This package concerns the object
    description of a sensor, such that it can be used in any of the
    targeted engines. A user can always extend the Sensor's config
    description to support other (custom) physics bridges. It also
    ensures that possible dependencies (e.g. hardware drivers), required
    for real world experiments, are met.

-   *eager_robot\_\[name_robot\]*: This package concerns the object
    description of a sensor, such that it can be used in any of the
    targeted s. A user can always extend the robot's config description
    to support other (custom) physics bridges. It also ensures that
    possible dependencies (e.g. hardware drivers), required for real
    world experiments, are met. In essence, this package structure is
    similar as the *eager_sensor* packages as both are considered to be
    *Objects* from EAGER's point of view.

-   *eager_process\_\[name_process\]*: This package concerns the process
    nodes for a specific type of observation or action. Though the nodes
    might be specific to a sensor/actuator, they are nonetheless engine
    agnostic, so they should work out-of-the-box with any of the
    targeted engines.

#### Included in current version

We have included three working examples in the current version of the
OpenDR toolkit. These examples make use of the *eager_core* package, the
physics bridges to Pybullet (*eager_bridge_pybullet*), and the ROS
object packages of the UR5e robot (*eager_robot_ur5e*) and Multisense
S21 3D camera sensor (*eager_sensor_multisense_s21*).

-   *example_multi_robot.launch*: This example demonstrates how easy it
    is to initialize an *EagerEnv* via composition in Pybullet. Such an
    environment would be useful for batch training with multiple robots
    at the same time. Also, this example uses images from a Multisense
    S32 3D camera *Object* to render rgb images.

-   *example_safe_actions.launch*: This example demonstrates how a
    safety layer can be added that ensures that the actions commanded by
    the agent are safe. This safety layer is implemented as a action
    process node from the ROS pakcage *eager_process_safe_actions*.
    Also, this example uses images from a Multisense S32 3D camera
    *Object* to render rgb images.

Due to limitations in the Webots simulator, we are unable to showcase
all functionality when choosing Webots as the engine. For example, the
current version of Webots does not allow the Webots bridge to extract
state information other than what is defined to be an observation.
Hence, also a state reset is not possible. Also, we have not yet been
able to cleanly perform synchronized steps of the engine in a
multi-robot setting (e.g., an environment with multiple UR5e's or an
environment with both a UR5e and a s21 camera), as the services of all
other robots are blocked when one of the robots pauses the simulator in
between steps. The next version (Webots R2021b) is expected to provide
functionality that will address these issues. Finally, Webots uses world
files (.wbt) to initialize the simulation world including all object.
Therefore, it does not allow the ad-hoc spawning of objects. As a work
around, we plan to develop a world file generator that allows the
generation of world files on-the-fly at initialization. Nevertheless, to
illustrate the ease with which EAGER can switch between engines, we've
included a separate example that uses the physics bridge to Webots
(*eager_bridge_webots*).

-   *example_switch_engine.launch*: In this example, we have a single
    UR5e robot that performs random actions that are processed to be
    safe. To switch engines in *example_switch_engine.py*, simply swap
    the *WebotsEngine* with the *PyBulletEngine*.

We leave the development of a sensor processing node and a real reset
node as future work. Specifically, we plan to have this functionality
implemented before the next review meeting. Similarly, we will also
continue to work on a physics bridge to fully support the Gazebo
simulator. Finally, we also leave the creation of a detailed tutorial on
how to create ROS packages for custom objects and physics bridges as
future work.

#### MyEnv(BaseEagerEnv)

A user can create a custom environment that inherits from the
BaseEagerEnv. In that case, the user must define the following
functions:

-   *\_\_init\_\_(name, engine)*: This is the constructor of the
    environment. It initializes the physics bridge via a launchfile
    based on the parameters specified in 'engine'. All ROS related
    services, topics, parameters, and nodes will be launched under the
    namespace specified by 'name'. After creating all robots and having
    optionally added processing nodes, the user must call
    *\_init_nodes(List\[Object\])* with a list containing all the
    Objects. This function initiates the services of each object, both
    on the MyEnv and EngineBridge side.

-   *step($a$)*: This method takes in the action $a$ as a dictionary
    that specifies the action for every object $a^i_r$, where index $i$
    denotes a specific actuator and $r$ the object. After setting all
    actions, the user must call *\_step()* to step the environment.
    After stepping the environment, the function returns the
    observations $o$ as a dictionary consisting of observations $o^i_r$,
    where index $i$ denotes a specific sensor and $r$ the object,
    together with a done flag, reward, and the object's state $s^i_r$.

-   *reset()*: This method resets every object to a specific state.
    First the desired reset state is set for each object, after which
    the \_reset() function is called. This signals the *EngineBridge*
    that a reset must be performed. It returns the observation $o$ after
    the reset has been performed.

-   *render()*: This method allows the user to render image observations
    of the simulation. It returns a 3-D numpy array. This function is
    only available if a Camera sensor is available in the environment.

-   *seed()*: This method sends publishes a message to the seed topic.
    Any node that deals with random numbers can subscribe to this Topic
    and seed accordingly. It returns the seed as in integer number.

-   *close()*: This method closes the environment. It returns None.

#### Object(BaseRosObject)

An *Object* inherits from the *BaseRosObject* base class. It has the
following functions:

-   *set_action($a_r$)*: This method takes as an argument the Object's
    actions $a_r$ as a dictionary. The dictionary contains the actions
    $a^i_r$ for all of the Object's Actuators. It sets the actions into
    a buffer with length 1. When the *EngineBridge* calls the Actuator's
    service during a step, it will receive the action from the buffer.

-   *get_obs()*: This method allows calls all the services of the
    EngineBridge that correspond to the Object's Sensors. It returns a
    dictionary of observations $o_r$.

-   *get_state()*: This method allows calls all the services of the
    EngineBridge that correspond to the Object's States. It returns a
    dictionary of states $s_r$.

-   *set_state($s_r$)*: This method takes as an argument the Object's
    desired state $s_r$ as a dictionary, to which we wish to reset the
    engine. The dictionary contains the states $a^i_r$ for all of the
    Object's States. It sets the states into a buffer with length 1.
    When the *EngineBridge* calls the State's service for a reset, it
    will receive the state from the buffer.

### Usage Example

In this section, we first demonstrate how the user can initialize an
EAGER environment. Then, we describe two methods on how the user can
define their own custom environment. It is important to note that,
before running any EAGER code, a roscore must be launched in a separate
process (i.e., in a separate terminal). EAGER environments can both be
launched within a ROS node, or from within Python code using a Python
interpreter. Users are encouraged to create ROS object packages for
their robots. Similarly, it is possible for the user to use their own
(custom) physics engine by creating a ROS package that has a
*MyEngineBridge* that inherits from the *PhysicsBridge* base class.
However, as we mentioned in a previous section, we leave the creation of
a detailed tutorial on how to create such ROS packages as future work.

#### Using EAGER

EAGER environments are designed to be agnostic to the physics engine
(i.e., Webots, Pybullet, real world). Given an EAGER environment, the
user must therefore specify at initialization what physics engine to
use. For this, the user can use a pre-existing parameter class that
accompanies every engine's ROS package. It defines how the chosen engine
should be initialized. Most parameters are optional such as the world
timestep and are initialized with a default value. However, some
parameters do not have a default value and must be provided by the user
(e.g., the world file in Webots). Note how file paths can be provided
with ROS substitution args (e.g. find) that EAGER resolves prior to
launching/loading.

``` {.python language="Python"}
engine = WebotsEngine(world='$(find opendr_example)/worlds/ur5e_cam.wbt')
engine = PyBulletEngine(world='plane.urdf')
```

The user can proceed the initialization with the chosen engine by simply
passing the parameter class to the environment's constructor, together
with a unique environment name. All ROS services, topics, parameters,
and other ROS nodes (e.g. Webots itself) will be launched within the
namespace of the provided name.

``` {.python language="Python"}
env = MyEnv(engine, name="my_env", **kwargs)
env = Flatten(env
env.seed(42)
```

With the environment initialized, the agent can interact with it as it
would with any typical OpenAI Gym environment.

``` {.python language="Python"}
obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
```

#### Defining an environment

The user can define a custom environment MyEnv by inheriting from
*BaseEagerEnv*. In the constructor, the user can create any object, by
specifying a unique name(space), the ROS package, and its type. All ROS
related things will be launched within the object's unique namespace.
Optionally, the user can specify object specific parameters, such as the
base pose or whether the baselink is fixed. In this example, we create a
UR5e robot and a camera object by referencing the respective ROS
packages under the namespaces 'ur5e1' and 'ms21', respectively.

In addition, we add preprocessing to the UR5e's joint actuators, which
checks that the agent's action commands are safe (i.e. collision-free).
Similar to physics engines, preprocessing nodes have a parameter class
that specify the required and optional arguments that must be passed to
the *add_preprocess()* function, together with the location of the
preprocess launchfile. In case the preprocess node requires observations
from an object (e.g. to check for collisions, you may require the
position of other objects) they can be provided as a list to the
'observations_from_objects' argument.

After defining all objects and preprocessing nodes, they must be passed
to the *\_init_nodes(\...)* function. This function then initializes all
the services, both on the *EagerEnv* and *EngineBridge* side.

``` {.python language="Python"}
class MyEnv(BaseEagerEnv):
    def __init__(self, engine, name="my_env"):
        super().__init__(engine, name=name)
    
        # Create ur5e object
        self.ur5e = Object.create('ur5e1', 'eager_robot_ur5e', 'ur5e')
    
        # Add preprocessing so that commanded actions are safe
        process_args = SafeActionsProcessor(...)
        self.ur5e.actuators['joints'].add_preprocess(
            launch_path='$(find eager_process_safe_actions)/launch/safe_actions.launch',
            launch_args=process_args.__dict__,
            observations_from_objects=[self.ur5e])
    
        # Create a camera object
        self.camera = Object.create('ms21', 'eager_sensor_multisense_s21', 'dual_cam')
    
        # Initialize all the services of the robots
        self._init_nodes([self.camera, self.ur5e])
    
        # Define the gym spaces
        self.observation_space = self.ur5e.observation_space
        self.action_space = self.ur5e.action_space
```

The step function can be defined as follows. First, all actions must be
set for each object. In this example, this only concerns the UR5e, as
the ms21 camera does not have any actuators. After setting the actions
for each object (that has actuators), the *\_step()* function must be
called. This function signals the *EngineBridge* that the engine can
perform a simulation timestep. The function blocks until the simulation
timestep on the *EngineBridge* side has finished. The *EngineBridge*
grabs the actions that were previously set by *MyEnv* by calling the
action services. After the *\_step* service returns in *MyEnv*, the
latest observation and state are retrieved by calling the observation
and state services of the *EngineBridge*.

``` {.python language="Python"}
def step(self, action):
    # Set actions before stepping
    self.ur5e.set_action(action)

    # Step the environment
    self._step()
    self.steps += 1

    # Get observations
    obs = self.ur5e.get_obs()
    info = self.ur5e.get_state()

    return obs, self._get_reward(obs), self._is_done(obs), info 
```

We define the step function as

``` {.python language="Python"}
def reset(self) -> object:
    # Set desired reset state
    reset_states = self.ur5e.state_space.sample()
    self.ur5e.reset(states=reset_states)

    # Reset the environment
    self._reset()

    # Return observations after reset
    return self.ur5e.get_obs()
```

Although most engines have the option to visualise simulations in a GUI,
turning it off often increases the training speed as it minimizes
unnecessary computations. To still be able to visually inspect the
agent's learned behavior over time, the user can define a render
function that uses a camera object to produce rgb images when called
upon.

``` {.python language="Python"}
def render(self, mode, **kwargs):
    # Use camera to render rgb images
    rgbd = self.camera.sensors['camera_right'].get_obs()
    return rgbd[:, :, :3]
```

#### EagerEnv

Many tasks are similar in the sense that the observation space and
action space of the environment is simply a composition of all
observations and actions of each robot. Defining a new environment for
every such task is not only a tedious process, but also unnecessary.
Hence, as an alternative to defining custom environments, we also
provide a useful environment called *EagerEnv*, that allows the user to
directly compose new environments on-the-fly by simply providing its
constructor with a list of objects, together with a custom
*\_reward_fn()* (and optionally a custom *is_done()* function). The gym
spaces for this environment are composed of the union of observation and
action spaces of all objects in the provided list. This demonstrates
EAGER's preference for composition over inheritance.

``` {.python language="Python"}
# Create a grid of ur5e robots
objects = []
grid = [-1.5, 0, 1.5]Ik
for x in grid:
    for y in grid:
        idx = len(objects)
        ur5e = Object.create('ur5e%d%d' % (x,y), 'eager_robot_ur5e', 'ur5e', position=[x, y, 0])
        objects.append(ur5e)

# Dummy reward fn - Here, we output a batch reward for each ur5e.
def reward_fn(obs):
    rwd = []
    for obj in obs:
        if 'ur5e' in obj:
            rwd.append(-(obs[obj]['joint_sensors'] ** 2).sum())
    return rwd

# Add a camera for rendering
cam = Object.create('ms21', 'eager_sensor_multisense_s21', 'dual_cam')
render_obs = cam.sensors['camera_right'].get_obs
objects.append(cam)

# Create environment
env = EagerEnv(engine=engine, name='multi_env', objects=objects, render_obs=render_obs, reward_fn=reward_fnd)
env = Flatten(env)
```
