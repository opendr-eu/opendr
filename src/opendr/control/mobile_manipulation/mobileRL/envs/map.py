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
import random
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Union, List, Tuple, Optional

from opendr.control.mobile_manipulation.mobileRL.envs.env_utils import quaternion_to_yaw
from opendr.control.mobile_manipulation.mobileRL.envs.simulator_api import GazeboAPI, DummySimulatorAPI, SpawnObject, \
    ObjectGeometry

SMALL_NUMBER = 1e-6


def resize_to_resolution(current_resolution: float, target_resolution: float, map):
    using_np = isinstance(map, np.ndarray)
    if using_np:
        map = Image.fromarray(map)
    size_orig = np.array(map.size)
    size_new = tuple((current_resolution / target_resolution * size_orig).astype(np.int))
    map = map.resize(size_new)
    if using_np:
        map = np.asarray(map)
    return map


class Map:
    def __init__(self,
                 floorplan: Union[Path, np.array],
                 orig_resolution: float,
                 map_frame_rviz: str,
                 inflation_radius: float,
                 local_map_size_meters: float = 3.0,
                 origin: Optional[Tuple[float, float, float]]=None):
        # m per cell (per pixel)
        self._resolution = 0.1
        self.map_path = floorplan if isinstance(floorplan, (str, Path)) else None
        # [W, H], resized to match the defined resolution so we will have the same size for the local maps
        self._floorplan_img_orig = self.load_map(floorplan,
                                                 current_resolution=orig_resolution,
                                                 target_resolution=self._resolution)
        self._floorplan_img = self._floorplan_img_orig.copy()
        self._map_W_meter = self._resolution * self._floorplan_img.width
        self._map_H_meter = self._resolution * self._floorplan_img.height
        # in meters
        if origin is None:
            # take the center as the origin
            origin = (self._map_W_meter / 2, self._map_H_meter / 2, 0)
        self._origin_W_meter, self._origin_H_meter, self._origin_Z_meter = origin

        sz = int(local_map_size_meters / self._resolution)
        self.output_size = (sz, sz)

        self._map_frame_rviz = map_frame_rviz

        # set to half of the robot base size (plus padding if desired)
        self.inflation_radius_meter = inflation_radius

    @staticmethod
    def load_map(map: Union[str, Path, np.array], current_resolution: float, target_resolution: float,
                 pad: bool = True) -> Image:
        """
        :param map: either a path to an image or a numpy array of shape [H x W]
        :param current_resolution:
        :param target_resolution:
        :return: a binary map with occupied==1, free==0
        """
        if isinstance(map, (Path, str)):
            assert Path(map).exists()
            # '1' is a binary map occupied / free
            # NOTE: PIL interprets '0' as black and '1' as white. So always use plt.imshow(np.asarray(img)) to visualise
            map = Image.open(map).convert('1')
            # invert so that black are obstacles (True) and non-black are free (False)
            map = Image.fromarray(~np.asarray(map))
        else:
            # expects the input array to be HxW, transposing it into WxH
            map = Image.fromarray(map.astype(np.bool))
        assert len(map.size) == 2, map.size

        # resize to be of target_resolution
        map = resize_to_resolution(current_resolution, target_resolution, map)

        # pad the map so the base doesn't run out of bound and always gets the collision penalty at the bounds
        if pad:
            # pad by 1 meter
            padding = round(1.0 / target_resolution)
            map = ImageOps.expand(map, border=padding, fill=1)

        return map

    def meter_to_pixels(self, xypose_meter: np.ndarray, round: bool = True):
        """
        In ROS the y-axis points upwards with origin at the center, in pixelspace it points downwards with origin in the
        top left corner
        """
        xypose_meter = xypose_meter.copy()
        xypose_meter[..., 0] += self._origin_W_meter
        xypose_meter[..., 1] = self._origin_H_meter - xypose_meter[..., 1]
        xypose_pixel = xypose_meter / self._resolution
        if round:
            xypose_pixel = np.round(xypose_pixel).astype(np.int)
        return xypose_pixel

    def pixels_to_meter(self, xypose_pixel: np.ndarray):
        xypose_meter = xypose_pixel * self._resolution
        xypose_meter[..., 0] -= self._origin_W_meter
        xypose_meter[..., 1] = self._origin_H_meter - xypose_meter[..., 1]
        return xypose_meter

    def draw_goal(self, *args, **kwargs):
        pass

    def draw_initial_base_pose(self, gripper_goal_wrist):
        return [0, 0, 0, 0, 0, 0, 1]

    def map_reset(self):
        self._floorplan_img = self._floorplan_img_orig.copy()

    def clear(self):
        pass

    @staticmethod
    def _plot_map(map, show: bool = False):
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(np.asarray(map).squeeze())
        if show:
            plt.show()
        return f, ax

    def plot_floorplan(self, show: bool = False, location=None):
        f, ax = self._plot_map(self._floorplan_img, show)
        if location is not None:
            location_pixels = self.meter_to_pixels(np.array(location)[..., :2])
            # NOTE: prob. not vectorised yet
            theta = quaternion_to_yaw(np.array(location)[..., 3:])
            # scale arrow length to a percentage of local map size
            c = 0.1 * np.array(self.output_size)
            ax.arrow(location_pixels[..., 0], location_pixels[..., 1],
                     c[0] * np.cos(theta), c[1] * -np.sin(theta),
                     color='cyan', width=1)
        return f, ax


class DummyMap(Map):
    def __init__(self):
        super(DummyMap, self).__init__(floorplan=np.zeros([2, 2]), orig_resolution=0.1, map_frame_rviz="map",
                                       inflation_radius=0.2)

    def in_collision(self, xy_meters) -> bool:
        return False

    def get_local_map(self, current_location):
        return np.zeros(0, 0)


class EmptyMap(Map):
    """An empty map starting at the origin and sampling goals in a distance around it"""

    def __init__(self,
                 map_frame_rviz: str,
                 inflation_radius: float,
                 initial_base_rng_x=(0, 0),
                 initial_base_rng_y=(0, 0),
                 initial_base_rng_yaw=(0, 0),
                 # W x H
                 size_meters: Tuple[float, float]=(11, 11),
                 resolution: float = 0.1):
        super(EmptyMap, self).__init__(
            floorplan=self.get_empty_floorplan(size_meters=size_meters, resolution=resolution),
            origin=None,
            orig_resolution=resolution,
            map_frame_rviz=map_frame_rviz,
            inflation_radius=inflation_radius)
        # base
        assert len(initial_base_rng_x) == len(initial_base_rng_y) == len(initial_base_rng_yaw) == 2
        self._initial_base_rng_x = initial_base_rng_x
        self._initial_base_rng_y = initial_base_rng_y
        # in radians
        self._initial_base_rng_yaw = initial_base_rng_yaw

    @staticmethod
    def get_empty_floorplan(size_meters: Tuple[float, float], resolution: float = 0.1):
        # returns an array H x W as Image.fromarray expects
        map = np.zeros([int(size_meters[1] / resolution), int(size_meters[0] / resolution)], dtype=bool)
        # wall around the map
        map[[0, -1], :] = True
        map[:, [0, -1]] = True
        return map

    def draw_initial_base_pose(self, gripper_goal_wrist):
        return [random.uniform(*self._initial_base_rng_x),
                random.uniform(*self._initial_base_rng_y),
                0, 0, 0,
                random.uniform(*self._initial_base_rng_yaw)]


class SceneMap(EmptyMap):
    def __init__(self,
                 world_type: str,
                 robot_frame_id: str,
                 inflation_radius: float,
                 initial_base_rng_x=(0, 0),
                 initial_base_rng_y=(0, 0),
                 initial_base_rng_yaw=(0, 0),
                 requires_spawn: bool = False,
                 fixed_scene_objects: Optional[List[SpawnObject]] = None):
        super(SceneMap, self).__init__(initial_base_rng_x=initial_base_rng_x,
                                       initial_base_rng_y=initial_base_rng_y,
                                       initial_base_rng_yaw=initial_base_rng_yaw,
                                       map_frame_rviz=robot_frame_id if world_type == 'sim' else 'map',
                                       inflation_radius=inflation_radius)
        if (world_type == "sim"):
            # still need gazebo to figure out the goals in some tasks
            if requires_spawn:
                self.simulator = GazeboAPI()
            else:
                self.simulator = DummySimulatorAPI(frame_id=robot_frame_id)
        elif world_type == "gazebo":
            self.simulator = GazeboAPI()
        elif world_type == "world":
            raise NotImplementedError()
        else:
            raise NotImplementedError(world_type)

        self.fixed_scene_objects = fixed_scene_objects or []

    def get_varying_scene_objects(self) -> List[SpawnObject]:
        return []

    def get_scene_objects(self) -> List[SpawnObject]:
        return self.get_varying_scene_objects() + self.fixed_scene_objects

    def _get_rectangle(self, x_meter, y_meter, width_meter, height_meter, angle_radian):
        x, y = self.meter_to_pixels(np.array([x_meter, y_meter], dtype=np.float))
        width, height = round(width_meter / self._resolution), round(height_meter / self._resolution)

        half_w = width / 2
        half_h = height / 2
        rect = np.array([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
        R = np.array([[np.cos(angle_radian), -np.sin(angle_radian)],
                      [np.sin(angle_radian), np.cos(angle_radian)]])
        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def _add_rectangle(self, x_meter, y_meter, width_meter, height_meter, angle_radian, floorplan=None):
        if floorplan is not None:
            draw = ImageDraw.Draw(floorplan)
        else:
            draw = ImageDraw.Draw(self._floorplan_img)
        rect = self._get_rectangle(x_meter=x_meter, y_meter=y_meter,
                                   width_meter=width_meter, height_meter=height_meter,
                                   angle_radian=angle_radian)
        draw.polygon([tuple(p) for p in rect], fill=1)

    def add_objects_to_floormap(self, objects: List[SpawnObject]):
        for obj in objects:
            if obj.world_object.geometry == ObjectGeometry.RECTANGLE:
                yaw = quaternion_to_yaw(obj.pose.orientation)
                self._add_rectangle(x_meter=obj.pose.position.x, y_meter=obj.pose.position.y,
                                    width_meter=obj.world_object.x, height_meter=obj.world_object.y,
                                    angle_radian=yaw)
            elif obj.world_object.geometry == ObjectGeometry.UNKNOWN:
                pass
            else:
                raise NotImplementedError()

    def map_reset(self):
        super().map_reset()
        self.simulator.delete_all_spawned()
        objects = self.get_scene_objects()
        self.simulator.spawn_scene_objects(objects)
        self.add_objects_to_floormap(objects)

    def clear(self):
        return self.simulator.clear()
