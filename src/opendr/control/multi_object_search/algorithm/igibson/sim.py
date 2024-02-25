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


import pybullet as p
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.objects.stateful_object import StatefulObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.constants import SemanticClass
from igibson.simulator import Simulator


class Sim(Simulator):
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
            self,
            gravity=9.8,
            physics_timestep=1 / 120.0,
            render_timestep=1 / 30.0,
            solver_iterations=100,
            mode="gui",
            image_width=128,
            image_height=128,
            vertical_fov=90,
            device_idx=0,
            render_to_tensor=False,
            rendering_settings=MeshRendererSettings(),
            vr_settings=VrSettings(),
            door_list=[]
    ):

        self.pb_id_to_sem_id_map = {}
        self.door_list = door_list
        super(Sim, self).__init__(
            gravity=gravity,
            physics_timestep=physics_timestep,
            render_timestep=render_timestep,
            solver_iterations=solver_iterations,
            mode=mode,
            image_width=image_width,
            image_height=image_height,
            vertical_fov=vertical_fov,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            rendering_settings=rendering_settings,
            vr_settings=vr_settings
        )

    def load_without_pybullet_vis(load_func):
        """
        Load without pybullet visualizer
        """

        def wrapped_load_func(*args, **kwargs):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            res = load_func(*args, **kwargs)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            return res

        return wrapped_load_func

    @load_without_pybullet_vis
    def import_ig_scene(self, scene):
        """
        Import scene from iGSDF class

        :param scene: iGSDFScene instance
        :return: pybullet body ids from scene.load function
        """
        self.pb_id_to_sem_id_map = {}

        assert isinstance(
            scene, InteractiveIndoorScene
        ), "import_ig_scene can only be called with InteractiveIndoorScene"
        if not self.use_pb_renderer:
            scene.set_ignore_visual_shape(True)
            # skip loading visual shape if not using pybullet visualizer

        new_object_ids = scene.load()
        self.objects += new_object_ids

        # after the above statement, the doors has been loaded and they now have some pybullet id.
        # run through the door object dictionary in igibson_interactive_scene in order to get thFse ids
        # door_ids = []

        start_label = 375

        i = 0
        for body_id, visual_mesh_to_material, link_name_to_vm in zip(
                new_object_ids, scene.visual_mesh_to_material, scene.link_name_to_vm
        ):
            use_pbr = True
            use_pbr_mapping = True
            shadow_caster = True
            physical_object = scene.objects_by_id[body_id]

            if scene.scene_source == "IG":
                if physical_object.category in ["walls", "floors", "ceilings"]:
                    use_pbr = False
                    use_pbr_mapping = False
                if physical_object.category == "ceilings":
                    shadow_caster = False

            if physical_object.name.startswith("door"):  # in self.door_list:
                class_id = start_label + i
                self.pb_id_to_sem_id_map[body_id] = class_id
                i += 1
            else:
                class_id = self.class_name_to_class_id.get(physical_object.category, SemanticClass.SCENE_OBJS)
            """
            if body_id in door_labels:
                class_id = door_labels[body_id]
            else:
                class_id = self.class_name_to_class_id.get(physical_object.category, SemanticClass.SCENE_OBJS)
            """
            self.load_articulated_object_in_renderer(
                body_id,
                class_id=class_id,
                visual_mesh_to_material=visual_mesh_to_material,
                link_name_to_vm=link_name_to_vm,
                use_pbr=use_pbr,
                use_pbr_mapping=use_pbr_mapping,
                shadow_caster=shadow_caster,
                physical_object=physical_object,
            )

        self.scene = scene

        # Load the states of all the objects in the scene.
        for obj in scene.get_objects():
            if isinstance(obj, ObjectMultiplexer):
                for sub_obj in obj._multiplexed_objects:
                    if isinstance(sub_obj, ObjectGrouper):
                        for group_sub_obj in sub_obj.objects:
                            for state in group_sub_obj.states.values():
                                state.initialize(self)
                    else:
                        for state in sub_obj.states.values():
                            state.initialize(self)
            elif isinstance(obj, StatefulObject):
                for state in obj.states.values():
                    state.initialize(self)

        return new_object_ids
