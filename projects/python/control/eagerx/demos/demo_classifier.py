#!/usr/bin/env python
# Copyright 2020-2023 OpenDR European Project
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
import os

# EAGERx imports
from eagerx import Object, Bridge, Node, initialize, log
from eagerx.core.graph import Graph
import eagerx.bridges.openai_gym as eagerx_gym
import eagerx_examples  # noqa: F401

# Import stable-baselines
import stable_baselines3 as sb


def example_classifier(name, eps, eval_eps, device, render=False):
    # Start roscore & initialize main thread as node
    initialize("eagerx", anonymous=True, log_level=log.INFO)

    # Define object
    pendulum = Object.make(
        "GymObject",
        "pendulum",
        sensors=["image", "observation", "reward", "done"],
        env_id="Pendulum-v0",
        rate=20,
        always_render=True,
        render_shape=[28, 28],
    )

    # Define PID controller & classifier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    classifier = Node.make("Classifier", "classifier", rate=20, cam_rate=20, data=dir_path + "/../data/with_actions.h5")
    pid = Node.make("PidController", "pid", rate=20, gains=[8, 1, 0], y_range=[-4, 4])

    # Define graph (agnostic) & connect nodes
    graph = Graph.create(nodes=[classifier, pid], objects=[pendulum])
    graph.connect(source=pendulum.sensors.reward, observation="reward")
    graph.connect(source=pendulum.sensors.done, observation="done")
    graph.connect(source=classifier.outputs.state, observation="state")
    # Connect Classifier
    graph.connect(source=classifier.outputs.state, target=pid.inputs.y)
    graph.connect(source=pendulum.sensors.image, target=classifier.inputs.image)
    # Connect PID
    graph.connect(action="yref", target=pid.inputs.yref)
    graph.connect(source=pid.outputs.u, target=pendulum.actuators.action)
    # Add rendering
    if render:
        graph.render(source=pendulum.sensors.image, rate=10, display=True)

    # Define bridge
    bridge = Bridge.make("GymBridge", rate=20)

    # Initialize Environment (agnostic graph +  bridge)
    env = eagerx_gym.EagerxGym(name=name, rate=20, graph=graph, bridge=bridge)
    if render:
        env.render(mode="human")

    # Initialize and train stable-baselines model
    model = sb.SAC("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=int(eps * 200))

    # Evaluate trained policy
    for i in range(eval_eps):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
    env.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--name", help="Name of the environment", type=str, default="example")
    parser.add_argument("--eps", help="Number of training episodes", type=int, default=200)
    parser.add_argument("--eval_eps", help="Number of evaluation episodes", type=int, default=20)
    parser.add_argument("--render", help="Toggle rendering", action="store_true")

    args = parser.parse_args()

    example_classifier(name=args.name, eps=args.eps, eval_eps=args.eval_eps, device=args.device, render=args.render)
