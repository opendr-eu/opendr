# EAGERx Demos

Engine Agnostic Gym Environment with Reactive extension (EAGERx) is a toolkit that will allow users to apply (deep) reinforcement learning for both simulated and real robots as well as combinations thereof.
Documentation is available [here](../../../docs/reference/eagerx.md).
The source code of EAGERx is available [here](https://github.com/eager-dev/eagerx)

**Prerequisites**: EAGERx requires ROS Noetic and Python 3.8 to be installed.

This folder contains minimal code usage examples that showcase some of EAGERx's features. 
Specifically the following examples are provided:
1. **[demo_full_state](demos/demo_full_state.py)**:  
   Here, we wrap the OpenAI gym within EAGERx.
   The agent learns to map low-dimensional angular observations to torques.
2. **[demo_pid](demos/demo_pid.py)**:   
   Here, we add a PID controller, tuned to stabilize the pendulum in the upright position, as a pre-processing node.
   The agent now maps low-dimensional angular observations to reference torques.
   In turn, the reference torques are converted to torques by the PID controller, and applied to the system.
3. **[demo_classifier](demos/demo_classifier.py)**:   
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

## Citing EAGERx

To cite EAGERx in publications:
```bibtex
@article{eagerx,
    author  = {van der Heijden, Bas and Luijkx, Jelle, and Ferranti, Laura and Kober, Jens and Babuska, Robert},
    title = {EAGER: Engine Agnostic Gym Environment for Robotics},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/eager-dev/eagerx}}
}
```
