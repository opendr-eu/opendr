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


from igibson.reward_functions.potential_reward import PotentialReward


class PotentialRewardClipped(PotentialReward):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialRewardClipped, self).__init__(config)
        # specific value depending on the robot used.
        # The Fetch robot can move faster in the direction of the cloest object as the Locobot
        # => Fetch clip value must the larger
        self.clip_value = config.get("pot_rew_clip_value", 0.2)

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = task.get_potential(env)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return min(reward, self.clip_value)
