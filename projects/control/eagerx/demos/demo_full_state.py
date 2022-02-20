#!/usr/bin/env python
# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

# EAGERx imports
from eagerx import Object, Bridge, initialize, log
from eagerx.core.graph import Graph
import eagerx.bridges.openai_gym as eagerx_gym
import eagerx_examples  # noqa: F401


def example_full_state(name, eps, eval_eps, device):
    # Start roscore & initialize main thread as node
    initialize("eagerx", anonymous=True, log_level=log.INFO)

    # Define object
    pendulum = Object.make("GymObject", "pendulum", gym_env_id="Pendulum-v0", gym_rate=20)

    # Define graph (agnostic) & connect nodes
    graph = Graph.create(objects=[pendulum])
    graph.connect(source=("pendulum", "sensors", "observation"), observation="observation", window=1)
    graph.connect(source=("pendulum", "sensors", "reward"), observation="reward", window=1)
    graph.connect(source=("pendulum", "sensors", "done"), observation="done", window=1)
    graph.connect(action="action", target=("pendulum", "actuators", "action"), window=1)

    # Define bridge
    bridge = Bridge.make("GymBridge", rate=20)

    # Initialize Environment (agnostic graph +  bridge)
    env = eagerx_gym.EagerGym(name=name, rate=20, graph=graph, bridge=bridge)

    # Use stable-baselines
    import stable_baselines3 as sb

    model = sb.SAC("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=int(eps * 200))

    # Evaluate trained policy
    for i in range(eval_eps):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    env.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--name", help="Name of the environment", type=str, default="example")
    parser.add_argument("--eps", help="Number of training episodes", type=int, default=200)
    parser.add_argument("--eval_eps", help="Number of evaluation episodes", type=int, default=20)

    args = parser.parse_args()

    example_full_state(name=args.name, eps=args.eps, eval_eps=args.eval_eps, device=args.device)
