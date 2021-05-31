# OpenDR Mobile Manipulation - Kinematic Feasibility

This folder contains the OpenDR Learner class for mobile manipulation tasks. This method uses reinforcement learning to train an agent that is able to control the base of a wheeled robot to ensure given end-effector motions remain kinematically feasible.

## Sources

The implementation is based on [Learning Kinematic Feasibility for Mobile Manipulation through Deep Reinforcement Learning](https://arxiv.org/abs/2101.05325).
The environment and code are refactored and modularised versions of the [originally published code](https://github.com/robot-learning-freiburg/kinematic-feasibility-rl).

The following ROS files located in `robots_world` are slight modifications of files originally provided by the following ROS packages:
- `hsr/hsr_analytical.launch`: [hsrb_moveit_config](https://github.com/hsr-project/hsrb_moveit_config)
- `pr2/pr2.urdf.xacro`: [pr2_description](http://wiki.ros.org/pr2_description)
- `pr2/pr2_analytical.launch`: [pr2_moveit_config](http://wiki.ros.org/pr2_moveit_config)
- `tiago/tiago_analytical.launch`, `tiago/modified_tiago.srdf.em`, `tiago/modified_tiago_pal-gripper.srdf`: tiago_moveit_config
- `tiago/modified_gripper.urdf.xacro`: pal_gripper_description
- `tiago/modified_wsg_gripper`: pal_wsg_gripper_description
- `tiago/tiago.urdf.xacro`, `tiago/upload.launch`: tiago__description

