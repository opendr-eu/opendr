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


from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class ObjectGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(ObjectGoalReward, self).__init__(config)
        self.object_values = [64, 32, 12, 102, 126, 140]  # ,12,12,12,12,12]
        self.min_pix = [29, 29, 29, 29, 29, 29, 29]  # ,29,29,29,29]
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """

        reward = -0.0025
        for i in task.uniq_indices:

            if task.wanted_objects[i] == 1:
                # 224x224x3
                masked_category = (env.global_map[:, :, 0] == self.object_values[i])
                num_pixel = masked_category.sum()

                if (num_pixel > int(self.min_pix[i])):
                    # constrain the area
                    success = l2_distance(env.robots[0].get_position()[:2], task.target_pos_list[i][:2]) < self.dist_tol
                    if success:
                        reward += 10.0
                        task.wanted_objects[i] = 0

                        env.global_map[masked_category] = env.mapping.colors['object_found']

                        if (task.current_target_ind == i):
                            task.reward_functions[0].reset(task, env)

        return reward

    def reset(self, task, env):
        pass
        # self.indices = np.argwhere(env.wanted_objects != 0)
