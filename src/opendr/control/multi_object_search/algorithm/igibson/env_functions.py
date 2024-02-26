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


from igibson.envs.env_base import BaseEnv
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import parse_config
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from opendr.control.multi_object_search.algorithm.igibson.fetch import Fetch_DD
from opendr.control.multi_object_search.algorithm.igibson.locobot import Locobot_DD
from opendr.control.multi_object_search.algorithm.igibson.sim import Sim


class BaseFunctions(BaseEnv):
    """
    Base Env class, follows OpenAI Gym interface
    Handles loading scene and robot
    Functions like reset and step are not implemented
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            render_to_tensor=False,
            device_idx=0,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """

        self.config = parse_config(config_file)

        if scene_id is not None:
            self.config["scene_id"] = scene_id

        self.keep_doors_list = self.mapping.map_settings[self.config["scene_id"]]['doors']
        self.remove_doors = self.config["remove_doors"]
        # in upper class
        self.mapping.load_miscellaneous_map(self.config["scene_id"])

        self.mode = self.config.get("mode", "headless")
        self.action_timestep = self.config.get("action_timestep", 1.0) / self.config.get("action_timestep_div", 10.0)
        self.physics_timestep = self.config.get("physics_timestep", 1.0) / self.config.get("physics_timestep_div",
                                                                                           120.0)
        self.test_demo = self.config.get("test_demo", False)
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)
        self.object_randomization_idx = 0
        self.num_object_randomization_idx = 10

        enable_shadow = self.config.get("enable_shadow", False)
        enable_pbr = self.config.get("enable_pbr", True)
        texture_scale = self.config.get("texture_scale", 1.0)

        # TODO: We currently only support the optimized renderer due to some issues with obj highlighting
        settings = MeshRendererSettings(
            enable_shadow=enable_shadow, enable_pbr=enable_pbr, msaa=False, texture_scale=texture_scale, optimized=True
        )

        self.simulator = Sim(
            mode=self.mode,
            physics_timestep=self.physics_timestep,
            render_timestep=self.action_timestep,
            image_width=self.config.get("image_width", 128),
            image_height=self.config.get("image_height", 128),
            vertical_fov=self.config.get("vertical_fov", 90),
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            rendering_settings=settings,
            door_list=self.keep_doors_list
        )
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another scene model
        This allows one to change the scene on the fly

        :param scene_id: new scene_id
        """
        self.config["scene_id"] = scene_id
        self.keep_doors_list = self.mapping.map_settings[scene_id]['doors']
        self.mapping.load_miscellaneous_map(scene_id)
        self.simulator.reload()
        self.load()

    def load(self):
        """
        Load the scene and robot
        """

        if self.config["scene"] == "gibson":
            scene = StaticIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
            )
            self.simulator.import_scene(scene, load_texture=self.config.get("load_texture", True))
        elif self.config["scene"] == "igibson":
            scene = InteractiveIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                trav_map_type=self.config.get("trav_map_type", "with_obj"),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get("should_open_all_doors", False),
                load_object_categories=self.config.get("load_object_categories", None),
                load_room_types=self.config.get("load_room_types", None),
                load_room_instances=self.config.get("load_room_instances", None),

            )
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get("_set_first_n_objects", -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_ig_scene(scene)

        if self.config["robot"] == "Fetch":
            robot = Fetch_DD(self.config)
        elif self.config["robot"] == "Locobot":
            robot = Locobot_DD(self.config)
        else:
            raise Exception("unknown robot type: {}".format(self.config["robot"]))

        self.scene = scene
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot)
