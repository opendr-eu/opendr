# Copyright 2020-2024 OpenDR European Project
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

import numpy as np
import time
import torch


def evaluation_rollout(policy, env, num_eval_episodes: int, verbose: bool = True,
                       name_prefix: str = '', name_scene: str = '', deterministic_policy: bool = False):

    name_prefix = f"{name_prefix + '_' if name_prefix else ''}MultiObjectEnv"

    episode_rewards, episode_lengths, episode_successes, episode_spls, collisions,  = [
        [] for _ in range(5)]
    env.reload_model(name_scene)
    env.last_scene_id = name_scene
    env.current_episode = 0

    with torch.no_grad():
        for ep in range(num_eval_episodes):

            t = time.time()
            done = False
            obs = env.reset()
            initial_geo_dist = env.task.initial_geodesic_length
            agent_geo_dist_taken = 0
            rewards, infos = [], []
            while not done:
                curr_position = env.robots[0].get_position()[:2]
                action, _states, aux_angle = policy.predict(obs, deterministic=deterministic_policy)
                action_mod = {"action": action, "aux_angle": aux_angle[0]}
                obs, reward, done, info = env.step(action_mod)

                rewards.append(reward)
                infos.append(info)
                episode_length = len(rewards)

                new_position = env.robots[0].get_position()[:2]
                _, geodesic_dist = env.scene.get_shortest_path(
                    env.task.floor_num,
                    curr_position,
                    new_position,
                    entire_path=False)
                curr_position = new_position
                agent_geo_dist_taken += geodesic_dist

            episode_rewards.append(np.sum(rewards))
            if infos[-1]['success']:
                episode_spls.append(initial_geo_dist / max(initial_geo_dist, agent_geo_dist_taken))
            else:
                episode_spls.append(0)
            episode_lengths.append(episode_length)
            collisions.append(env.collision_step)

            episode_successes.append(int(infos[-1]['success']))

            if verbose > 1:
                print(f"{name_prefix}: Eval episode {ep}: \
                    {(time.time() - t) / 60:.2f} minutes, {episode_length} steps. \
                    Success-rate: {np.mean(episode_successes)}.")

    metrics = {'return_undisc': np.mean(episode_rewards),
               'epoch_len': np.mean(episode_lengths),
               'success': np.mean(episode_successes),
               'spl': np.mean(episode_spls)}

    return episode_rewards, episode_lengths, metrics, name_prefix
