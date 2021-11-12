# Copyright 2020-2021 OpenDR European Project
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
import rospy
import time
import torch
from matplotlib import pyplot as plt

from opendr.control.mobile_manipulation.mobileRL.envs.env_utils import calc_disc_return
from opendr.control.mobile_manipulation.mobileRL.utils import create_env


def episode_is_success(nr_kin_fails: int, nr_collisions: int, goal_reached: bool) -> bool:
    return (nr_kin_fails == 0) and (nr_collisions == 0) and goal_reached


def evaluation_rollout(policy, env, num_eval_episodes: int, global_step: int, verbose: bool = True,
                       name_prefix: str = ''):
    name_prefix = f"{name_prefix + '_' if name_prefix else ''}{env.loggingname}"

    episode_rewards, episode_lengths, episode_returns, episode_successes, fails_per_episode, goal_reached, vel_norms = [
        [] for _ in range(7)]
    max_len = 100_000
    with torch.no_grad():
        for i in range(num_eval_episodes):
            t = time.time()
            done = False
            obs = env.reset()

            rewards, infos, actions = [], [], []
            while not done:
                action, state = policy.predict(obs, deterministic=True)

                obs, reward, done, info = env.step(action)

                rewards.append(reward)
                infos.append(info)
                episode_length = len(rewards)
                actions.append(action)

                if episode_length > max_len:
                    assert episode_length < max_len, f"EPISODE OF {episode_length} STEPS!"

            episode_rewards.append(np.sum(rewards))
            return_disc = calc_disc_return(rewards, gamma=policy.gamma)
            episode_returns.append(return_disc)
            episode_lengths.append(episode_length)
            goal_reached.append(infos[-1]['ee_done'])
            # kin fails are cumulative
            fails_per_episode.append(infos[-1]['nr_kin_failures'])
            episode_successes.append(
                episode_is_success(nr_kin_fails=fails_per_episode[-1], nr_collisions=0, goal_reached=goal_reached[-1]))

            unscaled_actions = [env._convert_policy_to_env_actions(a) for a in actions]
            vel_norms.append(np.mean([a[0] for a in unscaled_actions]))

            if (verbose > 1) or (env.get_world() != "sim"):
                rospy.loginfo(
                    f"{name_prefix}: Eval ep {i}: {(time.time() - t) / 60:.2f} minutes, {episode_length} steps. "
                    f"Ik failures: {fails_per_episode[-1]}. "
                    f"{sum(episode_successes)}/{i + 1} full success.")

    log_dict = {}
    if env._learn_vel_norm != -1:
        log_dict['vel_norm'] = np.mean(vel_norms)

    ik_fail_thresh = env._ik_fail_thresh
    fails_per_episode = np.array(fails_per_episode)
    metrics = {'return_undisc': np.mean(episode_rewards),
               'return_disc': np.mean(episode_returns),
               'epoch_len': np.mean(episode_lengths),
               f'ik_b{ik_fail_thresh}': np.mean(fails_per_episode <= ik_fail_thresh),
               'ik_b11': np.mean(fails_per_episode < 11),
               'ik_zero_fail': np.mean(fails_per_episode == 0),
               'ik_fails': np.mean(fails_per_episode),
               'goal_reached': np.mean(goal_reached),
               'success': np.mean(episode_successes),
               'global_step': global_step,
               'timesteps_total': global_step}
    rospy.loginfo("---------------------------------------")
    rospy.loginfo(f"T {global_step}, {name_prefix:} evaluation over {num_eval_episodes:.0f} episodes: "
                  f"Avg. return (undisc) {metrics[f'return_undisc']:.2f}, (disc) {metrics[f'return_disc']:.2f}, "
                  f"Avg failures {metrics[f'ik_fails']:.2f}, Avg success: {metrics['success']:.2f}")
    rospy.loginfo(
        f"IK fails: {metrics[f'ik_b{ik_fail_thresh}']:.2f}p < {ik_fail_thresh}, {metrics[f'ik_b11']:.2f}p < 11, "
        f"{metrics[f'ik_zero_fail']:.2f}p < 1")
    rospy.loginfo("---------------------------------------")

    log_dict.update({(f'{name_prefix}/{k}' if ('step' not in k) else k): v for k, v in metrics.items()})
    for k, v in log_dict.items():
        policy.logger.record(k, v)

    plt.close('all')
    return episode_rewards, episode_lengths, metrics, name_prefix


def evaluate_on_task(config, agent, eval_env_config, task: str, world_type: str):
    eval_env_config = eval_env_config.copy()
    eval_env_config['world_type'] = world_type
    env = create_env(eval_env_config,
                     task=task,
                     node_handle=eval_env_config["node_handle"],
                     flatten_obs=True)

    rospy.loginfo(f"Evaluating on task {env.taskname()} with {world_type} execution.")
    prefix = ''
    if world_type != 'sim':
        prefix += f'ts{config["time_step"]}_slow{config["slow_down_real_exec"]}'

    agent.eval(env, nr_evaluations=config["nr_evaluations"], name_prefix=prefix)
    env.clear()
