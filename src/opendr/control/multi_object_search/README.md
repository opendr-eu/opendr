# OpenDR Multi Object Search - Long Horizon Reasoning

This folder contains the OpenDR Learner class for Multi Object Search. This method uses reinforcement learning to train an agent that is able to control the base of a wheeled robot (Fetch and LoCoBot checkpoint provided) to locomote the agent through unseen environments.

## Sources

The implementation is based on [Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces](https://opendr.eu/wp-content/uploads/2022/07/Learning-Long-Horizon-Robot-Exploration-Strategies-for-Multi-Object-Search-in-Continuous-Action-Spaces.pdf).

The environment and code are refactored and modularised versions of the [originally published code](https://github.com/robot-learning-freiburg/Multi-Object-Search).

The following files located in `algorithm/igibson` are slight modifications of files originally provided by the following iGibson packages:
- `Locobot/Locobot.urdf`: [LoCoBot description](http://www.locobot.org/)
- `Fetch/fetch.urdf`: [Fetch description](https://fetchrobotics.com/fetch-mobile-manipulator/)
- `scenes/*`: [modified iGibson floor-maps](https://stanfordvl.github.io/iGibson/scenes.html?highlight=floor%20map)
- `algorithm/igibson/fetch.py`: [iGibson fetch asset](https://stanfordvl.github.io/iGibson/assets.html?highlight=fetch)
- `algorithm/igibson/locobot.py`: [iGibson locobot asset](https://stanfordvl.github.io/iGibson/assets.html?highlight=locobot)
- `algorithm/igibson/ycb_object_id.py`: [YCB dataset](https://stanfordvl.github.io/iGibson/assets.html?highlight=ycb)
- `algorithm/igibson/sim.py`: [iGibson simulator file](https://github.com/StanfordVL/iGibson/blob/master/igibson/simulator.py)


The following files located in `algorithm/SB3` are slight modifications of files originally provided by the following stable-baseline3 packages:
- `algorithm/SB3/buffer.py`: [SB3 buffer class](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
- `algorithm/SB3/policy.py`: [SB3 policy class]https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
- `algorithm/SB3/ppo.py`: [SB3 ppo class](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
- `algorithm/SB3/torch_layers.py`: [SB3 torch_layers class](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
- `algorithm/SB3/type_aliases.py`: [SB3 type_aliases class](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
- `algorithm/SB3/vec_env.py`: [SB3 vec_env class](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/common)
