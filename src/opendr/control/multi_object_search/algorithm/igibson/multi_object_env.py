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


import logging
from collections import OrderedDict

import cv2
import gym
import numpy as np
import pybullet as p
from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.utils.constants import MAX_CLASS_COUNT
from igibson.utils.utils import quatToXYZW
from scipy.special import softmax
from scipy.stats import circvar
from transforms3d.euler import euler2quat

from opendr.control.multi_object_search.algorithm.igibson.env_functions import BaseFunctions
from opendr.control.multi_object_search.algorithm.igibson.mapping_module import MappingModule
from opendr.control.multi_object_search.algorithm.igibson.multi_object_task import MultiObjectTask


class MultiObjectEnv(BaseFunctions):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            device_idx=0,
            render_to_tensor=False,
            automatic_reset=False,
            mix_sample=True
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        self.mapping = MappingModule(config_file, scene_id)

        self.mix_sample = mix_sample
        super(MultiObjectEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,

        )

        self.automatic_reset = automatic_reset

        self.queue_for_task_resampling = []
        self.episode_counter = 0
        self.scene_reset_counter = 0
        self.evaluate = self.config.get('evaluate', False)
        if self.evaluate:
            self.aux_episodic_prob = 1.0
            self.multiple_envs = False
            self.resample_task = False
        else:
            self.aux_episodic_prob = self.config.get('initial_aux_prob', 0.16)
            self.multiple_envs = self.config.get('multiple_envs', True)
            self.resample_task = self.config.get('resample_task', True)

        self.max_aux_episodic_prob = self.config.get('max_aux_episodic_prob', 0.72)

        self.SR_rate = []

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task

        self.task = MultiObjectTask(self)

        self.task_2_object = self.config.get("task_2_object", False)
        self.task_3_object = self.config.get("task_3_object", False)
        self.numb_sem_categories = self.config.get("sem_categories", 1)
        self.last_scene_id = self.config.get("scene_id", 'Rs_int')
        self.task.load_custom_objects(self)
        if not self.test_demo:
            self.task.load_door_material(self)
        else:
            self.task.load_door_proxy_material()

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("depth")

        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        observation_space = OrderedDict()
        self.last_action_obs = self.config.get('last_action_obs', False)
        self.use_rgb_depth = self.config.get('rgb_depth', False)
        self.downsample_size = self.config.get('global_map_size', 128)

        self.history_length_aux = self.config.get('history_length_aux', 10)
        self.numb_sem_categories = self.config.get("sem_categories", 1)

        additional_stuff = self.mapping.aux_bin_number + (self.history_length_aux * self.mapping.aux_bin_number)
        observation_space["task_obs"] = self.build_obs_space(
            shape=(self.numb_sem_categories + 1 + 2 + 1 + 1 + 2 + additional_stuff + 2,), low=-np.inf, high=np.inf)

        observation_space['image'] = gym.spaces.Box(low=0, high=255, shape=(
            3, self.mapping.cut_out_size[0], self.mapping.cut_out_size[1]), dtype=np.uint8)
        observation_space['image_global'] = gym.spaces.Box(low=0, high=255,
                                                           shape=(3, self.downsample_size, self.downsample_size),
                                                           dtype=np.uint8)

        self.observation_space = gym.spaces.Dict(observation_space)

        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment
        """
        super(MultiObjectEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """

        new_collision_links = []
        for item in collision_links:

            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:  # or item[3] in [9,6]:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[
                    4] in self.collision_ignore_link_a_ids:  # or (item[2] in [3]):
                continue

            if item[2] in self.task.remove_collision_links:
                continue

            new_collision_links.append(item)

        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step
        info["aux_episode"] = self.episodic_aux_prediction

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """

        # action['action'][1] *= -1
        self.robots[0].apply_action(action['action'])

        self.current_step += 1
        collision_links = self.run_simulation()

        self.collision_links = collision_links
        c = int(len(collision_links) > 0)

        # if c == 1:
        #    print("HAs collision")
        # else:
        #    print("no colliosion")
        self.collision_step += c

        self.coll_track.append(c)
        self.coll_track.pop(0)

        if len(self.prev_locations) >= self.history_length_aux:
            self.aux_angle_track.pop(0)
            self.prev_locations.pop(0)

        state = self.get_state(collision_links)

        ego1, ego2 = self.mapping.run_mapping(self, state, action)

        self.prev_locations.append(self.robots[0].robot_body.get_position()[:2])

        info = {}

        reward, info = self.task.get_reward(self, collision_links, action, info)
        if reward > 9.0:
            ego1, ego2 = self.mapping.run_mapping(self, state, action, only_ego_map=True)
        done, info = self.task.get_termination(self, collision_links, action, info)
        # task.step performs visualization, path length addition of last and new position and assigns new rob pos.
        self.task.step(self)
        self.populate_info(info)

        task_o = state['task_obs'].copy()

        linear_vel, angular_vel = task_o[2:]
        if task_o[1] < 0:
            task_o[1] += 2 * np.pi

        gt_bin = np.digitize([np.rad2deg(task_o[1])], self.mapping.angle_to_bins, right=True)

        gt_bin_vec = np.zeros(self.mapping.aux_bin_number) + np.random.uniform(0, 0.1)

        gt_bin_vec[gt_bin] += 1.0
        gt_bin_vec = softmax(gt_bin_vec)

        state = {}

        state['image'] = ego1.transpose(2, 0, 1)

        state['image_global'] = ego2.transpose(2, 0, 1)

        if self.episodic_aux_prediction:
            self.aux_angle_track.append(np.argmax(self.mapping.aux_action))  # <<<<<<<<<<<<<<<<
        else:
            self.aux_angle_track.append(task_o[1])

        var0 = np.var(np.array(self.prev_locations)[:, 0])
        var1 = np.var(np.array(self.prev_locations)[:, 1])

        # if(self.aux_task):
        state['task_obs'] = np.concatenate([gt_bin_vec, np.array(self.prev_aux_predictions).flatten(),
                                            [linear_vel, angular_vel, circvar(self.aux_angle_track), var0, var1],
                                            np.array([int(len(collision_links) > 0)]),
                                            np.array([np.array(self.coll_track).sum()]), action['action'],
                                            self.task.wanted_objects])

        self.prev_aux_predictions.pop(0)
        self.prev_aux_predictions.append(self.mapping.aux_action)

        info['true_aux'] = gt_bin[0]  # task_o[:2].copy()#<<<<<<<<<<<<<<<<

        if self.episodic_aux_prediction:  # or self.eval:

            state['task_obs'][:self.mapping.aux_bin_number] = self.mapping.aux_action

        if self.mapping.show_map:
            cv2.imshow("coarse", state['image'].transpose(1, 2, 0).astype(np.uint8))
            cv2.waitKey(1)

            cv2.imshow("global", state['image_global'].transpose(1, 2, 0).astype(np.uint8))
            cv2.waitKey(1)

            tmp = self.global_map.copy()

            # For Visualizing the angle on the static map.
            # w_ax = self.world2map(euler_mat.dot(self.mapping.aux_points[np.argmax(self.mapping.aux_action)].T).T
            # + camera_translation)
            # tmp[int(w_ax[1])-5:int(w_ax[1])+5,int(w_ax[0])-5:int(w_ax[0])+5] = self.aux_pred

            cv2.imshow("GLOBAL NO POLICY INPUT", tmp)
            cv2.waitKey(1)

        if done and not info['success'] and (self.episode_counter > 20 or self.evaluate) and (
                np.random.uniform() < 0.3 or self.evaluate):
            img_cpy = self.global_map.copy()
            open_ind = np.argwhere(self.task.wanted_objects == 1)
            img_cpy[int(self.mapping.rob_pose[1]) - 4:int(self.mapping.rob_pose[1]) + 4,
                    int(self.mapping.rob_pose[0]) - 4:int(self.mapping.rob_pose[0]) + 4] = self.mapping.colors['circle']
            for i in open_ind:
                tr_ps = self.task.target_pos_list[int(i)]
                tr_ps = self.mapping.world2map(tr_ps)
                img_cpy[int(tr_ps[1]) - 2:int(tr_ps[1]) + 2, int(tr_ps[0]) - 2:int(tr_ps[0]) + 2] = np.array(
                    [5, 128, 128])

            cv2.imwrite('pics/{}/img_{}_{}_{}_{}_colls_{}_{}'.format(self.last_scene_id, self.task.wanted_objects,
                                                                     self.current_step, info['success'],
                                                                     self.current_episode, self.collision_step, '.png'),
                        img_cpy)

        # Resampling task
        if done:  # and self.episodic_aux_prediction:
            self.SR_rate.append(info['success'])

        if (self.episode_counter > 25 and done and not info[
                'success'] and self.resample_task):  # IT WAS 25 BEFORE RETRAIN
            self.queue_for_task_resampling.append((self.task.initial_pos, self.task.initial_orn,
                                                   self.task.target_pos_list, self.task.initial_wanted,
                                                   self.last_scene_id))

        return state, reward, done, info

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.episode_counter += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []
        self.aux_pred_reset = np.zeros(self.mapping.aux_bin_number) + (1 / self.mapping.aux_bin_number)
        self.aux_angle_track = []
        self.prev_locations = []
        self.coll_track = [0.0] * self.history_length_aux
        self.prev_aux_predictions = [np.zeros(self.mapping.aux_bin_number)] * self.history_length_aux

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """

        if len(self.SR_rate) > 32:
            self.SR_rate.pop(0)

        self.episodic_aux_prediction = np.random.uniform() < self.aux_episodic_prob

        if self.mapping.aux_prob_counter > 4 and self.episode_counter > 25 and np.mean(
                np.array(self.SR_rate)) > 0.55 and not self.evaluate:
            self.aux_episodic_prob = min(self.max_aux_episodic_prob, self.aux_episodic_prob + 0.02)
            self.mapping.aux_prob_counter = 0

        if self.scene_reset_counter > 4 and self.multiple_envs and self.mix_sample:
            scene_id = np.random.choice(['Rs_int', 'Wainscott_1_int', 'Benevolence_0_int', 'Beechwood_1_int'])
            if self.last_scene_id != scene_id:
                self.reload_model(scene_id)
                self.scene_reset_counter = 0
                self.last_scene_id = scene_id

        self.scene_reset_counter += 1
        self.mapping.aux_prob_counter += 1

        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset_scene(self)
        self.task.reset_agent(self)

        if self.test_demo:

            # Elementary test on the Demo-scene Rs (not to confuse with Rs_int)
            self.land(self.robots[0], [0, 0, 0], [0, 0, 0])
            orn = np.array([0, 1, 1.5])
            positions_cracker = [
                [0.4, -1.0, 0.0],
                [0.4, -2.0, 0.0],
                [0.4, -3.0, 0.0],
                [0.95, -3.1, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0]]
            self.task.target_pos_list = np.array(positions_cracker)
            for i, cracker_list in enumerate(self.task.interactive_objects):
                pos = positions_cracker[i]
                self.land(cracker_list[0], pos, orn)
                pos2 = pos.copy()
                pos2[0] += 0.195
                self.land(cracker_list[1], pos2, orn)
                pos3 = pos.copy()
                pos3[0] -= 0.195
                self.land(cracker_list[2], pos3, orn)

        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()
        self.mapping.reset(self)
        self.global_map = np.zeros((self.mapping.map_size[0], self.mapping.map_size[1], 3), dtype=np.uint8) * 255

        if self.test_demo:
            self.global_map[:, :, :] = self.mapping.colors['walls']

        self.seg_mask = (state['seg'] * 255).astype(int)

        ego1, ego2 = self.mapping.run_mapping(self, state, None)

        task_o = state['task_obs']

        state = {}

        state['image'] = ego1.transpose(2, 0, 1)
        state['image_global'] = ego2.transpose(2, 0, 1)

        gt_bin_vec = np.zeros(self.mapping.aux_bin_number) + (1 / self.mapping.aux_bin_number)

        state['task_obs'] = np.concatenate(
            [gt_bin_vec, np.array(self.prev_aux_predictions).flatten(), [task_o[2], task_o[3], 0.0, 0.0, 0.0],
             np.array([0.0]), np.array([0.0]), np.array([0.0, 0.0]), self.task.wanted_objects])  # <<<<<<<<<<<<<<<<

        return state
