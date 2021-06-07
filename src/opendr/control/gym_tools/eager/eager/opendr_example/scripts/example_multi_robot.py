#!/usr/bin/env python3

# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ROS packages required
import rospy
from eager_core.eager_env import EagerEnv
from eager_core.objects import Object
from eager_core.wrappers.flatten import Flatten
from eager_bridge_pybullet.pybullet_engine import PyBulletEngine

import numpy as np
import pybullet_data
from stable_baselines3 import PPO

if __name__ == '__main__':

    rospy.init_node('example_safe_actions', anonymous=True, log_level=rospy.WARN)

    # Define the engine
    engine = PyBulletEngine(world='%s/%s.urdf' % (pybullet_data.getDataPath(), 'plane'), no_gui='false')

    # Create a grid of ur5e robots
    objects = []
    grid = [-1.5, 0, 1.5]
    for x in grid:
        for y in grid:
            idx = len(objects)
            objects.append(Object.create('ur5e%d' % idx, 'eager_robot_ur5e', 'ur5e', position=[x, y, 0]))

    # Dummy reward function - Here, we output a batch reward for each ur5e.
    # Anything can be defined here. All observations for each object in "objects" is included in obs
    def reward_fn(obs):
        rwd = []
        for obj in obs:
            if 'ur5e' in obj:
                rwd.append(-(obs[obj]['joint_sensors'] ** 2).sum())
        return rwd

    # Add a camera for rendering
    cam = Object.create('ms21', 'eager_sensor_multisense_s21', 'dual_cam')
    objects.append(cam)

    # Create environment
    env = EagerEnv(engine=engine, objects=objects, name='multi_env', render_obs=cam.sensors['camera_right'].get_obs, max_steps=100, reward_fn=reward_fn)
    env = Flatten(env)

    env.seed(42)

    rospy.loginfo("Training starts")

    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=100000)

    env.close()
