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


import logging

import numpy as np
import pybullet as p
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
from igibson.utils.utils import restoreState

from opendr.control.multi_object_search.algorithm.igibson.object_goal_rew import ObjectGoalReward
from opendr.control.multi_object_search.algorithm.igibson.object_goal_ter import ObjectGoal
from opendr.control.multi_object_search.algorithm.igibson.potential_reward_clip import PotentialRewardClipped
from opendr.control.multi_object_search.algorithm.igibson.ycb_object_id import YCBObject_ID


class MultiObjectTask(PointNavFixedTask):
    """
    Point Nav Random Task
    The goal is to navigate to a random goal position
    """

    def __init__(self, env):
        super(MultiObjectTask, self).__init__(env)

        self.target_dist_min = self.config.get("target_dist_min", 1.0)
        self.target_dist_max = self.config.get("target_dist_max", 10.0)
        self.num_tar_objects = self.config.get("tar_objects", 6.0)
        self.polar_to_geodesic = self.config.get("polar_to_geodesic", False)

        self.replace_objects = self.config.get(
            "replace_objects", True) if not self.config.get('evaluate', False) else False

        self.resample_episode_prob = self.config.get("resample_episode_prob", 0.15)

        self.test_demo = self.config.get("test_demo", False)
        if self.test_demo:
            # Accelerate demo tests
            self.termination_conditions[1].max_step = 500

        self.current_target_ind = 0
        self.prev_target_ind = 0
        self.num_categories = self.config.get("sem_categories", 1)
        self.queue_sample = False
        self.queue = []
        self.pre_load_failures = False

        if self.config.get("scene_id", 'Benevolence_0_int') in ["Benevolence_0_int", "Rs_int"]:
            self.object_distance = 1.0
        else:
            self.object_distance = 2.0

        # self.object_distance = 2.0
        self.failed = []
        if self.config.get("clip_potential_rew", False):
            self.reward_functions[0] = PotentialRewardClipped(self.config)

        self.reward_functions[-1] = ObjectGoalReward(self.config)

        self.termination_conditions[-1] = ObjectGoal(self.config)

    def reset_custom_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """

        if (self.queue_sample):
            print("Sampled objects from queue!")

            orn = np.array([0, 1, 1.5])

            for i, obj_list in enumerate(self.interactive_objects):
                state_id = p.saveState()
                pos = self.target_pos_list[i]

                env.land(obj_list[0], pos, orn)

                pos2 = pos.copy()
                pos2[0] += 0.195
                env.land(obj_list[1], pos2, orn)

                pos3 = pos.copy()
                pos3[0] -= 0.195
                env.land(obj_list[2], pos3, orn)
                p.removeState(state_id)
            return True

        max_trials = 100
        self.target_pos_list = []
        for obj_list in self.interactive_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)

                enough_distance_to_other_objects = True
                for already_pos in self.target_pos_list:
                    dist = l2_distance(pos, already_pos)
                    if dist < self.object_distance:
                        enough_distance_to_other_objects = False
                        break
                if not enough_distance_to_other_objects:
                    continue

                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    self.initial_pos[:2],
                    pos[:2],
                    entire_path=False
                )

                orn = np.array([0, 1, 1.5])

                reset_success1 = env.test_valid_position(obj_list[0], pos, orn)

                pos2 = pos.copy()
                pos2[0] += 0.195

                reset_success2 = env.test_valid_position(obj_list[1], pos2, orn)

                pos3 = pos.copy()
                pos3[0] -= 0.195
                # deactivate reset position for the other two side objects to accelerate training
                reset_success3 = env.test_valid_position(obj_list[2], pos3, orn)
                restoreState(state_id)
                reset_success = reset_success1 and reset_success2 and reset_success3
                if reset_success and (self.target_dist_min < dist):  # < self.target_dist_max):

                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")
                print("Scene:", env.last_scene_id)
                p.removeState(state_id)
                return False

            self.target_pos_list.append(pos)
            env.land(obj_list[0], pos, orn)
            env.land(obj_list[1], pos2, orn)
            env.land(obj_list[2], pos3, orn)
            p.removeState(state_id)
        return True

    def load_custom_objects(self, env):
        """
        Load interactive objects (YCB objects)

        :param env: environment instance
        :return: a list of interactive objects
        """
        self.interactive_objects = []
        object_paths = []

        self.object_distance = env.mapping.map_settings[env.last_scene_id]['object_dist']

        for _ in range(self.num_categories):
            object_paths.append('003_cracker_box')

        object_ids = []  # [80,81,82]
        for i in range(self.num_categories):
            object_ids.append(80 + i)  # 80

        self.ob_cats = [20400, 20655, 20910, 21165, 21420, 21675]
        self.cats_to_ind = {20400: 0, 20655: 1, 20910: 2, 21165: 3, 21420: 4, 21675: 5}

        self.remove_collision_links = []
        self.target_pos_list = []
        for i in range(self.num_categories):
            obj1 = YCBObject_ID(object_paths[i])
            obj2 = YCBObject_ID(object_paths[i])
            obj3 = YCBObject_ID(object_paths[i])

            env.simulator.import_object(obj1, class_id=object_ids[i])
            env.simulator.import_object(obj2, class_id=object_ids[i])
            env.simulator.import_object(obj3, class_id=object_ids[i])

            self.interactive_objects.append([obj1, obj2, obj3])

            self.remove_collision_links.append(obj1.bid)
            self.remove_collision_links.append(obj2.bid)
            self.remove_collision_links.append(obj3.bid)

        print("COLLISIONS LINKS:", self.remove_collision_links)
        # input()

    def load_door_proxy_material(self):
        self.forbidden_door_sem_ids = []
        self.door_sem_ids = []

    def load_door_material(self, env):

        self.door_sem_ids = np.arange(375, 390) * 255
        self.forbidden_door_sem_ids = []

        self.door_cat_to_ind = {}

        # dist_between_doors = env.mapping.map_settings[env.last_scene_id]['door_dist']
        keep_door_list = env.mapping.map_settings[env.last_scene_id]['doors']
        forbidden_door_list = env.mapping.map_settings[env.last_scene_id]['forbidden_doors']
        self.forbidden_door_ids = []
        self.removed_doors = {}
        self.all_door_ids = []
        self.door_dict = {}
        self.door_pos_list = []
        self.door_key_to_index = {}
        self.door_index_to_key = {}
        self.already_opened = {}

        self.name_to_sem_id_map = {}

        ind = 0
        for obj in env.scene.objects_by_category["door"]:
            key = obj.name

            door_id = obj.body_ids[0]

            if key in forbidden_door_list:
                sem_id = env.simulator.pb_id_to_sem_id_map[door_id]  # name_to_sem_id_map[key]
                self.forbidden_door_ids.append(door_id)

                self.forbidden_door_sem_ids.append(sem_id * 255)

            if key not in keep_door_list:

                self.removed_doors[door_id] = obj

            else:

                sem_id = env.simulator.pb_id_to_sem_id_map[door_id]
                self.all_door_ids.append(door_id)
                door_pos_orient = p.getBasePositionAndOrientation(door_id)
                pos_door = door_pos_orient[0]
                self.door_pos_list.append(pos_door)
                self.door_cat_to_ind[sem_id * 255] = ind
                ind += 1

        # print("----",self.door_cat_to_ind)
        self.original_door_mapping = self.door_cat_to_ind.copy()
        self.already_dilated_doors = np.zeros(ind)

    def reset_doors(self, env):
        env.scene.open_all_doors()
        for d_id in self.removed_doors.keys():

            self.removed_doors[d_id].main_body_is_fixed = False
            self.removed_doors[d_id].set_base_link_position_orientation(
                [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])

        for door_id in self.all_door_ids:

            env.scene.open_one_obj(door_id, mode="max")

            if door_id in self.forbidden_door_ids:
                env.scene.open_one_obj(door_id, mode="zero")

            else:
                # jointFrictionForce = 1
                for joint_id in range(p.getNumJoints(door_id)):
                    # p.setJointMotorControl2(pp[0], pp[1], p.POSITION_CONTROL, force=jointFrictionForce)
                    # set the parameters rediciously high in order to ensure that they can't MOVE !
                    p.changeDynamics(door_id, joint_id, mass=9999999.0, lateralFriction=0.1, spinningFriction=0.1,
                                     rollingFriction=0.1, frictionAnchor=True)

            p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

        # re-iniliaze the colors of the doors
        for d_key in self.door_cat_to_ind:
            self.door_cat_to_ind[d_key] = np.random.choice(env.mapping.num_door_color_indices)

        if len(self.all_door_ids) > 3:
            self.already_dilated_doors = np.zeros_like(self.already_dilated_doors)
        else:
            self.already_dilated_doors = np.zeros(4)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != "gui":
            return

        shortest_path = None
        for i in self.uniq_indices:  # for i in range(self.num_categories):
            if self.current_target_ind == i and self.wanted_objects[i] == 1:
                target = self.target_pos_list[i]
                self.initial_pos_vis_obj.set_position(self.initial_pos)
                self.target_pos_vis_obj.set_position(target)
                source = env.robots[0].get_position()[:2]
                shortest_path, _ = env.scene.get_shortest_path(self.floor_num, source, target[:2], entire_path=True)

        if shortest_path is None:
            return

        if env.scene.build_graph:

            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[i][0], shortest_path[i][1], floor_height])
                )
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))

    # This method is based on point_nav_fixed_task
    def get_shortest_path_mod(self, env, from_initial_pos=False, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]

        waypoints_ = []
        lowest_dist = np.inf

        for i in self.uniq_indices:  # range(self.num_categories):#self.current_indices:#
            if self.wanted_objects[i] == 1:
                target = self.target_pos_list[i][:2]
                waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                       entire_path=entire_path)
                if geodesic_dist < lowest_dist:
                    waypoints_ = waypoints
                    lowest_dist = geodesic_dist
                    self.current_target_ind = i

        return waypoints_, lowest_dist

    # overwritten method from point_nav_fixed_task
    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        waypoints, geodesic_dist = self.get_shortest_path_mod(env)
        self.current_waypoints = waypoints
        return geodesic_dist

    def get_initial_geodesic_length(self, env):
        # First find neasrest Object
        all_distances = []
        source = self.initial_pos[:2]
        already_visited = []
        while True:
            geodesic_dists = [np.inf] * self.num_categories
            waypoints_list = [[]] * self.num_categories
            base_case = True
            for i in range(self.num_categories):
                if i in already_visited:
                    continue
                target = self.target_pos_list[i][:2]
                if self.wanted_objects[i] == 1:
                    waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                           entire_path=True)
                    waypoints_list[i] = waypoints
                    geodesic_dists[i] = geodesic_dist
                    base_case = False

            if base_case:
                break
            nearest_object_from_source_ind = np.argmin(geodesic_dists)
            all_distances.append(geodesic_dists[nearest_object_from_source_ind])
            already_visited.append(nearest_object_from_source_ind)
            source = self.target_pos_list[i][:2]

        self.initial_geodesic_length = sum(all_distances)

    # overwritten method from point_nav_fixed_task
    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        if (self.polar_to_geodesic):
            ind = self.current_waypoints.shape[0] // 3
            self.next_waypoint = np.array([self.current_waypoints[ind][0], self.current_waypoints[ind][1], 1.0])
            task_obs = self.global_to_local(env, self.next_waypoint)[:2]
            if self.goal_format == "polar":
                task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))
            linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
            # angular velocity along the z-axis
            angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
            # if(self.task_obs_dim == 4):
            task_obs = np.append(task_obs, [linear_velocity, angular_velocity])

        else:

            task_obs = self.global_to_local(env, self.target_pos_list[self.current_target_ind])[:2]
            if self.goal_format == "polar":
                task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

            # linear velocity along the x-axis
            linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
            # angular velocity along the z-axis
            angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
            if (self.task_obs_dim == 4):
                task_obs = np.append(task_obs, [linear_velocity, angular_velocity])
            else:
                task_obs = np.array(
                    [linear_velocity, angular_velocity])  # np.append(task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """

        self.queue_sample = False

        if ((np.random.uniform() < self.resample_episode_prob and len(
                env.queue_for_task_resampling) > 0) or self.pre_load_failures):

            if self.pre_load_failures:
                initial_pos, initial_orn, tar_pos_list, init_obs, _ = self.queue[-1]
                self.queue = np.delete(self.queue, len(self.queue) - 1, 0)
                self.init_obs = init_obs
                self.target_pos_list = tar_pos_list
                self.queue_sample = True
            else:
                for i in range(len(env.queue_for_task_resampling)):
                    initial_pos, initial_orn, tar_pos_list, init_obs, scene_id = env.queue_for_task_resampling.pop(0)
                    # check whether current scene id is the one used in sample
                    if env.last_scene_id == scene_id:
                        self.init_obs = init_obs
                        self.target_pos_list = tar_pos_list
                        self.queue_sample = True
                        print(f"Took Sample from Queue !!! SCENE: {scene_id}")
                        return initial_pos, initial_orn
                    env.queue_for_task_resampling.append((initial_pos, initial_orn, tar_pos_list, init_obs, scene_id))
        if self.queue_sample:
            return initial_pos, initial_orn

        self.queue_sample = False
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)

        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        return initial_pos, initial_orn

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """

        self.floor_num = env.scene.get_random_floor()
        super(MultiObjectTask, self).reset_scene(env)
        for cracker_ob in self.interactive_objects:
            cracker_ob[0].set_base_link_position_orientation(
                [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
            cracker_ob[1].set_base_link_position_orientation(
                [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
            cracker_ob[2].set_base_link_position_orientation(
                [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        if not self.test_demo:
            self.reset_doors(env)

    def sample_target_object(self):
        self.wanted_objects = np.zeros(self.num_categories)
        if self.test_demo:
            self.indices = np.random.choice(np.arange(self.num_categories), size=self.num_tar_objects,
                                            replace=False)
        else:
            self.indices = np.random.choice(np.arange(self.num_categories), size=self.num_tar_objects,
                                            replace=self.replace_objects)

        self.current_category = np.array([int(self.indices[0])])
        # if 5 in self.indices:
        #    self.indices = np.delete(self.indices,5,0)
        self.wanted_objects[self.indices] = 1.0
        self.num_cat_in_episode = (np.unique(self.indices)).shape[0]
        self.uniq_indices = np.unique(self.indices)

        # self.current_indices = self.uniq_indices.copy()
        self.initial_wanted = self.wanted_objects.copy()

        if (self.queue_sample):
            self.wanted_objects = self.init_obs
            self.uniq_indices = np.argwhere(self.wanted_objects != 0)[:, 0]

        print(
            f"New Episode wants {self.wanted_objects} object with indices {self.indices} \
            number of categories for episode {self.num_cat_in_episode}")

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 120

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):

            initial_pos, initial_orn = self.sample_initial_pose_and_target_pos(env)

            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn)
            # and env.test_valid_position(env.robots[0], target_pos)
            restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        succ = False
        for i in range(max_trials):
            succ = self.reset_custom_objects(env)
            if succ:
                break

        self.sample_target_object()

        self.get_initial_geodesic_length(env)

        super(MultiObjectTask, self).reset_agent(env)
