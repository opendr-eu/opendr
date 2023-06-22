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


import cv2
import numpy as np
import pybullet as p
from igibson.utils.utils import parse_config
from scipy.special import softmax
from transforms3d.euler import euler2mat


class MappingModule():

    def __init__(self, config, scene_id):

        self.config = parse_config(config)
        self.load_miscellaneous_1()
        self.load_miscellaneous_2()
        if scene_id is not None:
            self.config["scene_id"] = scene_id
        self.load_miscellaneous_map(self.config.get("scene_id", 'Rs_int'))
        self.offset_for_cut = 150
        self.aux_points = []
        self.aux_bin_number = self.config.get("num_bins", 12)
        step_size = 360 / self.aux_bin_number
        self.angle_to_bins = np.arange(step_size, 360 + step_size, step_size)
        deg_steps = np.arange(0, 360, step_size)
        for i, deg in enumerate(deg_steps):
            ax = self.pol2cart(0.5, np.deg2rad(deg))
            ax = np.array([ax[0], ax[1], 0.0])
            self.aux_points.append(ax)

        self.door_positions = np.array([[], [], [], []])
        self.downsample_size = self.config.get('global_map_size', 128)
        self.circle_size = self.config.get("circle_size", 6)
        self.test_demo = self.config.get("test_demo", False)

    def world2map(self, xy):
        if (len(xy.shape) > 1):
            gpoints = np.array([self.offset, self.offset, self.offset]) + np.round(
                (xy + self.grid_offset) / self.grid_spacing)
            return gpoints

        else:
            x = (xy[0] + self.grid_offset[0]) / self.grid_spacing[0]
            y = (xy[1] + self.grid_offset[1]) / self.grid_spacing[1]

        return [np.round(x) + self.offset, np.round(y) + self.offset]

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def run_mapping(self, env, sensors, action, no_ego_map=False, only_ego_map=False):

        self.sim_rob_position = env.robots[0].robot_body.get_position()

        self.camera = env.robots[0].parts['eyes'].get_pose()
        # camera = env.robots[0]._links['eyes'].get_orientation()

        camera_translation = self.camera[:3]
        self.seg_mask = (sensors['seg'] * 255).astype(int)

        depth_map = (sensors['depth'] * (self.depth_high - self.depth_low) + self.depth_low)
        depth_map[sensors['depth'] > self.depth_high] = 0.0
        depth_map[sensors['depth'] < self.depth_low] = 0.0

        camera_angles = p.getEulerFromQuaternion(self.camera[3:])

        euler_mat = euler2mat(camera_angles[0], -camera_angles[1], camera_angles[2])

        self.rob_pose = self.world2map(np.array([self.sim_rob_position[0], self.sim_rob_position[1]]))

        if only_ego_map:
            return self.affine4map(env, env.robots[0].get_rpy()[2], camera_angles[2], action, euler_mat,
                                   camera_translation)

        point_cloud = self.pointcloud(depth_map)

        w = (self.seg_mask == 1020).squeeze()
        point_cloud_walls = point_cloud[w, :]

        point_cloud_walls = euler_mat.dot(point_cloud_walls.T).T + camera_translation
        point_cloud_walls = self.world2map(point_cloud_walls).astype(np.uint16)

        f = (self.seg_mask == 1275).squeeze()
        point_cloud_floor = point_cloud[f, :]

        point_cloud_floor = euler_mat.dot(point_cloud_floor.T).T + camera_translation

        point_cloud_floor = self.world2map(point_cloud_floor).astype(np.uint16)

        path_indices = env.global_map[..., 2] == self.colors['trace'][2]
        try:
            env.global_map[point_cloud_floor[:, 1], point_cloud_floor[:, 0]] = self.colors['free_space']
            env.global_map[point_cloud_walls[:, 1], point_cloud_walls[:, 0]] = self.colors['walls']
            env.global_map[path_indices] = self.colors['trace']

            objects = (np.unique(self.seg_mask)).astype(int)
            # objects = (np.unique(sensors['seg'])).astype(int)

            self.draw_object_categories(env, objects, point_cloud, euler_mat, camera_translation)

            # curr_sim_pose = camera_translation[0], camera_translation[1], camera_angles[2]

            # curr_sim_pose = self.camera[0],self.camera[1],camera_angles[2]
            self.rob_pose = self.world2map(np.array([self.camera[0], self.camera[1]]))

            env.global_map[int(self.rob_pose[1]) - 2:int(self.rob_pose[1]) + 2,
                           int(self.rob_pose[0]) - 2:int(self.rob_pose[0]) + 2] = self.colors['trace']

        except Exception as e:
            print("An exception occurred Mapping", e)
            print("CURRENT POSITION:", self.sim_rob_position[:2], "\n Orientation:", self.camera,
                  "\n rest:", camera_translation)

        if not no_ego_map:
            return self.affine4map(env, env.robots[0].get_rpy()[2], camera_angles[2], action, euler_mat,
                                   camera_translation)

    def reset(self, env):
        self.shuffle_indices = np.random.permutation(len(self.colors['door_colors']))

        self.door_colors = self.original_door_colors[self.shuffle_indices]
        # print("Door ShuFFle:",self.shuffle_indices)
        self.grid_offset = self.map_settings[env.last_scene_id]['grid_offset']
        self.grid_spacing = self.map_settings[env.last_scene_id]['grid_spacing']
        self.offset = self.map_settings[env.last_scene_id]['offset']
        self.aux_action = env.aux_pred_reset
        self.load_miscellaneous_map(env.last_scene_id)

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        left = np.abs(np.minimum(0, y1))
        right = np.maximum(y2 - img.shape[0], 0)
        top = np.abs(np.minimum(0, x1))
        bottom = np.maximum(x2 - img.shape[1], 0)
        img = np.pad(img, ((left, right), (top, bottom), (0, 0)), mode="constant")

        y1 += left
        y2 += left
        x1 += top
        x2 += top
        return img, x1, x2, y1, y2

    def crop_fn(self, img: np.ndarray, center, output_size):
        h, w = np.array(output_size, dtype=int)
        x = int(center[0] - w / 2)
        y = int(center[1] - h / 2)

        y1, y2, x1, x2 = y, y + h, x, x + w
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[y1:y2, x1:x2]

    def affine4map(self, env, rot_agent, rot_camera, action=None, euler_mat=None, camera_translation=None):

        img_copy = env.global_map.copy()

        if action is not None:

            self.aux_action = softmax(action['aux_angle'])

            p_ax = self.world2map(euler_mat.dot(np.array(self.aux_points).T).T + camera_translation)
            for i, py in enumerate(p_ax):
                img_copy[int(py[1]) - 2:int(py[1]) + 2, int(py[0]) -
                         2:int(py[0]) + 2, 1::] = self.colors['aux_pred'][1::]
                img_copy[int(py[1]) - 2:int(py[1]) + 2, int(py[0]) - 2:int(py[0]) + 2, 0] =\
                    self.colors['aux_pred'][0] * self.aux_action[i]

        pos = self.rob_pose

        cropped_map = self.crop_fn(img_copy, center=pos, output_size=(
            self.cut_out_size2[0] + self.offset_for_cut, self.cut_out_size2[1] + self.offset_for_cut))

        w, h, _ = cropped_map.shape
        center = (h / 2, w / 2)
        M = cv2.getRotationMatrix2D(center, np.rad2deg(rot_agent) + 90.0, 1.0)
        ego_map = cv2.warpAffine(cropped_map, M, (h, w),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))  # flags=cv2.INTER_AREA,INTER_NEAREST

        cv2.circle(
            img=ego_map,
            center=(
                (self.cut_out_size2[0] + self.offset_for_cut) // 2, (self.cut_out_size2[1] + self.offset_for_cut) // 2),
            radius=int(self.circle_size),
            color=(int(self.colors['circle'][0]), int(self.colors['circle'][1]), int(self.colors['circle'][2])),
            thickness=-1,
        )

        ego_map_local = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
                                     output_size=(self.cut_out_size[0], self.cut_out_size[1]))
        ego_map = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
                               output_size=(self.cut_out_size2[0], self.cut_out_size2[1]))
        ego_map_global = cv2.resize(ego_map, (self.downsample_size, self.downsample_size),
                                    interpolation=cv2.INTER_NEAREST)

        return ego_map_local, ego_map_global

    def pointcloud(self, depth):
        # fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        depth = depth.squeeze()
        rows, cols = depth.shape

        hfov = self.cam_fov / 360. * 2. * np.pi
        fx = rows / (2. * np.tan(hfov / 2.))

        vfov = 2. * np.arctan(np.tan(hfov / 2) * cols / cols)
        fy = cols / (2. * np.tan(vfov / 2.))

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0)  # & (depth < 255)
        z = np.where(valid, depth, 0.0)
        x = np.where(valid, z * (c - (rows / 2)) / fx, 0)
        y = np.where(valid, z * (r - (cols / 2)) / fy, 0)
        return np.dstack((z, -x, y))

    def draw_object_categories(self, env, objects_current_frame, point_cloud, euler_mat, camera_translation):
        # print(objects_current_frame)

        for ob in objects_current_frame:
            # 0 is outer space
            if (ob == 1020 or ob == 1275 or ob == 0):
                continue

            point_cloud_category = point_cloud[(self.seg_mask == ob).squeeze(), :]
            point_cloud_category = euler_mat.dot(point_cloud_category.T).T + camera_translation
            point_cloud_category = self.world2map(point_cloud_category).astype(np.uint16)

            if ob in env.task.ob_cats or ob == 1530:
                pass
            # sink
            elif (ob == 81600):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['sink']
            # bed now door
            elif (ob in env.task.forbidden_door_sem_ids):  # == 31110):#6375): 6375 was originally the bed class
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['forbidden_door']
            elif (ob in env.task.door_sem_ids):
                ind = env.task.door_cat_to_ind[ob]
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['door_colors'][ind]
            # sofa
            elif (ob == 82365):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors[
                    'sofa']  # category_console_table
            elif (ob == 99960):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['walls']
            else:
                if self.test_demo:
                    env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['free_space']
                else:
                    env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['sofa']

            # semantic categories
            if ob in env.task.ob_cats:
                ind = env.task.cats_to_ind[ob]
                if env.task.wanted_objects[ind] == 0:
                    # pass
                    env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.colors['object_found']
                else:
                    env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = \
                        self.colors['object_colors'][ind]

    def load_miscellaneous_2(self):
        # Images have BGR format
        # Sofa colors get assigned to sofa category and also for "unknown" category,
        # since the original unknown colors had too many similarities with desired object categories.
        self.colors = {
            'walls': np.array([0, 255, 0]),
            'free_space': np.array([255, 0, 0]),
            'trace': np.array([164, 0, 255]),
            'circle': np.array([0, 128, 255]),
            'sofa': np.array([95, 190, 45]),
            'object_found': np.array([249, 192, 203]),
            'sink': np.array([13, 66, 220]),
            'forbidden_door': np.array([77, 12, 128]),
            'aux_pred': np.array([192, 25, 79]),
            'object_colors': [
                np.array([64, 64, 64]), np.array([32, 152, 196]), np.array([12, 48, 96]),
                np.array([102, 32, 77]), np.array([126, 55, 133]), np.array([140, 109, 84])],
            'door_colors': np.array([
                np.array([81, 255, 222]), np.array([91, 255, 144]),
                np.array([117, 255, 106]), np.array([199, 255, 27])])}

        self.aux_prob_counter = 0

        self.num_door_color_indices = np.arange(self.config.get('num_door_colors', 4))
        self.original_door_colors = self.colors['door_colors'].copy()

        self.depth_high = self.config.get('depth_high', 2.5)
        self.depth_low = self.config.get('depth_low', 0.5)

        self.show_map = self.config.get('show_map', False)

        self.history_length = self.config.get('history_length', 1)

        self.cam_fov = self.config.get('vertical_fov', 79.0)
        self.initial_camera_pitch = 0.0

    def load_miscellaneous_map(self, scene_id):
        map_settings_size = self.map_settings[scene_id]['map_size']
        self.map_size = (map_settings_size, map_settings_size)  # 164 neu 142

        self.map_size_2 = (128, 128)
        # self.proxy_task = args.proxy

        self.cut_out_size = (84, 84)
        self.x_off1 = self.map_size[0] - self.cut_out_size[0]
        self.x_off2 = int(self.map_size[0] - (self.x_off1 // 2))
        self.x_off1 = self.x_off1 // 2

        self.y_off1 = self.map_size[1] - self.cut_out_size[1]
        self.y_off2 = int(self.map_size[1] - (self.y_off1 // 2))
        self.y_off1 = self.y_off1 // 2

        # greater map
        self.cut_out_size2 = (420, 420)
        self.x_off11 = self.map_size[0] - self.cut_out_size2[0]
        self.x_off22 = int(self.map_size[0] - (self.x_off11 // 2))
        self.x_off11 = self.x_off11 // 2

        self.y_off11 = self.map_size[1] - self.cut_out_size2[1]
        self.y_off22 = int(self.map_size[1] - (self.y_off11 // 2))
        self.y_off11 = self.y_off11 // 2

    def load_miscellaneous_1(self):  # load_miscellaneous_FABI

        self.grid_res = self.config.get('grid_res', 0.033)
        self.map_settings = {}

        self.map_settings['Benevolence_1_int'] = {'grid_offset': np.array([5.5, 10.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 1.4,
                                                  'doors': ['door_52', 'door_54', 'door_55'],
                                                  'forbidden_doors': ['door_52'], "map_size": 450}  # 1.2
        self.map_settings['Benevolence_2_int'] = {'grid_offset': np.array([5.5, 10.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 1.7,
                                                  'doors': ["door_35", "door_41", "door_43"], 'forbidden_doors': [],
                                                  "map_size": 450}  # 1.6
        self.map_settings['Beechwood_0_int'] = {'grid_offset': np.array([13.5, 8.5, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.0,
                                                'doors': ['door_93', 'door_109', 'door_97', 'door_98', 'door_99',
                                                          'door_101', 'door_102'],
                                                'forbidden_doors': ['door_93', 'door_109'], "map_size": 625}  # 2.5w

        self.map_settings['Pomaria_1_int'] = {'grid_offset': np.array([15.0, 7.0, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 2.0,
                                              'doors': ['door_65', 'door_70', 'door_72', 'door_73'],
                                              'forbidden_doors': ['door_65', 'door_70'], "map_size": 550}
        self.map_settings['Pomaria_2_int'] = {'grid_offset': np.array([7.5, 7.5, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 2.0, 'doors': ["door_29", "door_32"],
                                              'forbidden_doors': [], "map_size": 450}

        self.map_settings['Merom_1_int'] = {'grid_offset': np.array([10.0, 7.0, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 2.0,
                                            'doors': ['door_74', 'door_93', 'door_85', 'door_86', 'door_87', 'door_88',
                                                      'door_89'], 'forbidden_doors': ['door_74', 'door_93'],
                                            "map_size": 650}

        self.map_settings['Wainscott_0_int'] = {'grid_offset': np.array([8.5, 8.0, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.0,
                                                'doors': ['door_126', 'door_128', 'door_132', 'door_135', "door_134",
                                                          "door_136", "door_137"],
                                                'forbidden_doors': ['door_126', 'door_128', 'door_132', 'door_135'],
                                                "map_size": 750}  # <--- massive map

        self.map_settings['Merom_0_int'] = {'grid_offset': np.array([3.3, 2.2, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 2.0, 'door_dist': 2.0,
                                            'doors': ['door_63', 'door_64', 'door_67', 'door_60'],
                                            'forbidden_doors': ['door_60'], "map_size": 400}  #

        self.map_settings['Benevolence_0_int'] = {'grid_offset': np.array([5.5, 9.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 1.0, 'door_dist': 1.4,
                                                  'doors': ['door_9', 'door_12', 'door_13', 'door_11'],
                                                  'forbidden_doors': ['door_9', 'door_12', 'door_13'],
                                                  "map_size": 450}  #

        self.map_settings['Pomaria_0_int'] = {'grid_offset': np.array([15.0, 6.5, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 2.0, 'door_dist': 1.4,
                                              'doors': ['door_41', 'door_42', 'door_44', 'door_46', 'door_40'],
                                              'forbidden_doors': ['door_41', 'door_42'], "map_size": 550}  #

        self.map_settings['Wainscott_1_int'] = {'grid_offset': np.array([8.0, 8.0, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.0, 'door_dist': 1.4,
                                                'doors': ['door_86', 'door_89', 'door_92', 'door_93', 'door_95'],
                                                'forbidden_doors': [],
                                                "map_size": 700}

        self.map_settings['Rs_int'] = {'grid_offset': np.array([6.0, 5.5, 15.1]),
                                       'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                       'object_dist': 1.4, 'doors': ['door_54', 'door_52'],
                                       'forbidden_doors': ['door_54'], "door_dist": 1.0, "map_size": 450}  #

        self.map_settings['Ihlen_0_int'] = {'grid_offset': np.array([5.5, 3.0, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 2.0, 'door_dist': 1.4,
                                            'doors': ['door_42', 'door_46', 'door_47'], 'forbidden_doors': ['door_42'],
                                            "map_size": 450}

        self.map_settings['Beechwood_1_int'] = {'grid_offset': np.array([11.5, 8.5, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.0, 'door_dist': 3.1,
                                                'doors': ['door_80', 'door_81', 'door_83', 'door_87', 'door_88',
                                                          'door_89', 'door_93'], 'forbidden_doors': [],
                                                "map_size": 600}  #

        self.map_settings['Ihlen_1_int'] = {'grid_offset': np.array([7.0, 4.5, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 2.0, 'door_dist': 1.4,
                                            'doors': ['door_86', 'door_91', 'door_99', 'door_103', 'door_108'],
                                            'forbidden_doors': ['door_86', 'door_91'],
                                            "map_size": 500}

        # Test-demo Settings
        self.map_settings['Rs'] = {
            'grid_offset': np.array([6.0, 5.5, 15.1]),
            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
            'object_dist': 1.4, 'doors': ['door_54', 'door_52'],
                                'forbidden_doors': ['door_54'], "door_dist": 1.0, "map_size": 450}
