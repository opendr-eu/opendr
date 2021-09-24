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

import random
import rospy
import time
from collections import namedtuple
from enum import IntEnum
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetModelState, SetModelState, SetModelConfiguration, \
    SetModelConfigurationRequest
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from typing import Iterable

from opendr.control.mobile_manipulation.mobileRL.envs.env_utils import publish_marker, clear_all_markers


class ObjectGeometry(IntEnum):
    """Shapes that we know how to project as obstacles onto the map during training"""
    UNKNOWN = 0
    RECTANGLE = 1


GazeboObject = namedtuple("GazeboObject", "database_name x y z geometry")


class WorldObjects:
    """
    name in gazebo database, x, y, z dimensions
    see e.g. sdfs here: https://github.com/osrf/gazebo_models/blob/master/table/model.sdf
    """
    # our own, non-database objects
    muesli2 = GazeboObject('muesli2', 0.05, 0.15, 0.23, ObjectGeometry.RECTANGLE)
    kallax2 = GazeboObject('Kallax2', 0.415, 0.39, 0.65, ObjectGeometry.RECTANGLE)
    kallax = GazeboObject('Kallax', 0.415, 0.39, 0.65, ObjectGeometry.RECTANGLE)
    reemc_table_low = GazeboObject('reemc_table_low', 0.75, 0.75, 0.41, ObjectGeometry.RECTANGLE)


SpawnObject = namedtuple("SpawnObject", "name world_object pose frame")


class SimulatorAPI:
    def __init__(self):
        pass

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        for obj in spawn_objects:
            self._spawn_model(obj.name, obj.world_object, obj.pose, obj.frame)

    def _spawn_model(self, name: str, obj: GazeboObject, pose: Pose, frame='world'):
        raise NotImplementedError()

    def get_model(self, name: str, relative_entity_name: str):
        raise NotImplementedError()

    def set_model(self, name: str, pose: Pose):
        raise NotImplementedError()

    def delete_model(self, name: str):
        raise NotImplementedError()

    def delete_all_spawned(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    @staticmethod
    def get_link_state(link_name: str):
        raise NotImplementedError()

    def set_joint_angle(self, model_name: str, joint_names: list, angles: list):
        raise NotImplementedError()


class DummySimulatorAPI(SimulatorAPI):
    """
    Dummy so we can use the same API for the analytical env.
    Atm not used for any practical purpose as we can already visualise obstacles through the floorplan.
    """

    def __init__(self, frame_id: str):
        super().__init__()
        self._frame_id = frame_id
        # requires to have a rospy node initialised! (Not a given for ray remote actors).
        self._verbose = False

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        if not self._verbose:
            return

        for obj in spawn_objects:
            if obj.world_object.geometry == ObjectGeometry.RECTANGLE:
                publish_marker("obstacles",
                               marker_pose=obj.pose,
                               marker_scale=[obj.world_object.x, obj.world_object.y, obj.world_object.z],
                               marker_id=random.randint(1000, 100000),
                               frame_id=self._frame_id,
                               geometry="cube")

    def delete_all_spawned(self):
        if self._verbose:
            clear_all_markers(self._frame_id)

    def clear(self):
        if self._verbose:
            clear_all_markers(self._frame_id)


class GazeboAPI(SimulatorAPI):
    def get_model_template(self, model_name: str):
        return f"""\
        <sdf version="1.6">
            <world name="default">
                <include>
                    <uri>model://{model_name}</uri>
                </include>
            </world>
        </sdf>"""

    def __init__(self, time_out=10):
        super().__init__()
        # https://answers.ros.org/question/246419/gazebo-spawn_model-from-py-source-code/
        # https://github.com/ros-simulation/gazebo_ros_pkgs/pull/948/files
        print("Waiting for gazebo services...")
        rospy.wait_for_service("gazebo/delete_model", timeout=time_out)
        rospy.wait_for_service("gazebo/spawn_sdf_model", timeout=time_out)
        rospy.wait_for_service('/gazebo/get_model_state', timeout=time_out)
        self._delete_model_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self._spawn_model_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        self._get_model_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
        self._set_model_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)
        self._reset_simulation_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self._spawned_models = []

    def _pause_physics(self):
        return self._pause_physics_srv()

    def _unpause_physics(self):
        return self._unpause_physics_srv()

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        self._pause_physics()
        for obj in spawn_objects:
            self._spawn_model(obj.name, obj.world_object, obj.pose, obj.frame)
        self._unpause_physics()

    def _spawn_model(self, name: str, obj: GazeboObject, pose: Pose, frame='world'):
        product_xml = self.get_model_template(obj.database_name)

        self._spawned_models.append(name)
        info = self._spawn_model_srv(name, product_xml, "", pose, frame)

        while not self.get_model(name, "world").success:
            info = self._spawn_model_srv(name, product_xml, "", pose, frame)
            time.sleep(0.1)
            print(f"Waiting for model {name} to spawn in gazebo")

        return info

    def get_model(self, name: str, relative_entity_name: str):
        return self._get_model_srv(name, relative_entity_name)

    def delete_model(self, name: str):
        self._delete_model_srv(name)
        while self.get_model(name, "world").success:
            rospy.loginfo(f"Waiting to delete model {name}")
            self._delete_model_srv(name)
            time.sleep(0.1)
        self._spawned_models.remove(name)

    def delete_all_spawned(self):
        self._pause_physics()
        while self._spawned_models:
            m = self._spawned_models[0]
            self.delete_model(m)
        self._unpause_physics()
        time.sleep(0.2)

    def reset_world(self):
        print("RESET WORLD MIGHT NOT WORK CORRECTLY ATM")
        return self._reset_simulation_srv()

    def clear(self):
        self._pause_physics()
        self.delete_all_spawned()
        self._unpause_physics()
        # self.reset_world()
        self._delete_model_srv.close()
        self._spawn_model_srv.close()
        self._get_model_srv.close()
        self._set_model_srv.close()
        self._reset_simulation_srv.close()

    @staticmethod
    def get_link_state(link_name: str):
        """NOTE: need to initialise a rospy node first, o/w will hang here!"""
        msg = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=10)
        return msg.pose[msg.name.index(link_name)]

    def set_joint_angle(self, model_name: str, joint_names: list, angles: list):
        self._pause_physics()
        # assert 0 <= angle <= np.pi / 2, angle
        assert len(joint_names) == len(angles)
        set_model_configuration_srv = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        req = SetModelConfigurationRequest()
        req.model_name = model_name
        # req.urdf_param_name = 'robot_description'
        req.joint_names = joint_names  # list
        req.joint_positions = angles  # list
        res = set_model_configuration_srv(req)
        assert res.success, res
        self._unpause_physics()
        return res
