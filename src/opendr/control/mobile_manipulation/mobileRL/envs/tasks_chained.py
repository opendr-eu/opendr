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

import copy
import numpy as np
import os
import random
import time
import zipfile
from functools import partial
from geometry_msgs.msg import Point, Pose, Quaternion
from pathlib import Path
from pybindings import GMMPlanner, multiply_tfs
from typing import List
from urllib.request import urlretrieve

from opendr.control.mobile_manipulation.mobileRL.envs.eeplanner import LinearPlannerWrapper, GMMPlannerWrapper
from opendr.control.mobile_manipulation.mobileRL.envs.env_utils import pose_to_list, list_to_pose
from opendr.control.mobile_manipulation.mobileRL.envs.map import SceneMap
from opendr.control.mobile_manipulation.mobileRL.envs.mobile_manipulation_env import MobileManipulationEnv
from opendr.control.mobile_manipulation.mobileRL.envs.simulator_api import WorldObjects, SpawnObject
from opendr.control.mobile_manipulation.mobileRL.envs.tasks import BaseTask, TaskGoal, GripperActions
from opendr.engine.constants import OPENDR_SERVER_URL


def download_gmm_models():
    """Download the imitation learning models used by the GMMPlanner"""
    module_root = Path(__file__).parent.parent.parent

    if not (module_root / 'GMM_models' / 'GMM_grasp_KallaxDrawer.csv').exists():
        zip_path = module_root / 'GMM_models.zip'
        url = os.path.join(OPENDR_SERVER_URL, "control", "mobile_manipulation", 'GMM_models.zip')
        urlretrieve(url=url, filename=zip_path)
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            zip_ref.extractall(str(module_root))
        os.remove(zip_path)


class BaseChainedTask(BaseTask):
    SUBGOAL_PAUSE = 2

    @property
    def loggingname(self):
        name = super().loggingname
        if hasattr(self.map, 'obstacle_configuration') and (self.map.obstacle_configuration != 'none'):
            name = name.replace(self.taskname(), f'{self.taskname()}{self.map.obstacle_configuration}')
        return name

    def __init__(self, env: MobileManipulationEnv, map: SceneMap, default_head_start: float,
                 close_gripper_at_start: bool = True):
        super(BaseChainedTask, self).__init__(env=env, initial_joint_distribution="rnd", map=map,
                                              default_head_start=default_head_start,
                                              close_gripper_at_start=close_gripper_at_start)

        download_gmm_models()
        self._motion_model_path = Path(__file__).parent.parent.parent / "GMM_models"
        assert self._motion_model_path.exists(), self._motion_model_path

        self._current_goal = 0
        # will be set at every reset
        self._goals = []

    def grasp(self):
        wait_for_result = (self.SUBGOAL_PAUSE == 0)
        self.env.close_gripper(0.0, wait_for_result)

    def draw_goal(self) -> List[TaskGoal]:
        raise NotImplementedError()

    def get_goal_objects(self) -> List[SpawnObject]:
        return []

    def reset(self):
        # NOTE: OVERRIDING ANY PREVIOUS OBJECTS OF THE MAP. COULD PROBABLY BE HANDLED A BIT MORE GENERAL
        self.env.map.fixed_scene_objects = self.get_goal_objects()
        self._current_goal = 0
        self._goals = self.draw_goal()
        first_goal = self._goals[self._current_goal]
        assert self._success_thres_dist == first_goal.success_thres_dist, "Can't handle this case yet, would lead to " \
                                                                          "different thresholds for ee and robot"
        assert self._success_thres_rot == first_goal.success_thres_rot, "Can't handle this case yet, would lead to " \
                                                                        "different thresholds for ee and robot"
        return super().reset(first_goal)

    def _episode_cleanup(self):
        self.env.open_gripper(wait_for_result=False)
        self.map.clear()

    def step(self, action):
        obs, reward, done, info = self.env.step(action=action)

        if done and info['ee_done']:
            end_action = self._goals[self._current_goal].end_action
            if end_action == GripperActions.GRASP:
                self.grasp()
            elif end_action == GripperActions.OPEN:
                self.env.open_gripper(wait_for_result=False)

            if self._current_goal < len(self._goals) - 1:
                new = self._goals[self._current_goal + 1]
                ee_planner = new.ee_fn(gripper_goal_tip=new.gripper_goal_tip,
                                       gripper_goal_wrist=self.env.tip_to_gripper_tf(new.gripper_goal_tip),
                                       head_start=new.head_start,
                                       map=self.env.map)
                obs = self.env.set_ee_planner(ee_planner=ee_planner,
                                              success_thres_dist=new.success_thres_dist,
                                              success_thres_rot=new.success_thres_rot)
                self._current_goal += 1
                done = False

        # ensure nothing left attached to the robot / the robot could spawn into / ...
        if done:
            self._episode_cleanup()

        return obs, reward, done, info

    def clear(self):
        self._episode_cleanup()
        super(BaseChainedTask, self).clear()


class ObstacleConfigMap(SceneMap):
    def __init__(self,
                 world_type: str,
                 robot_frame_id: str,
                 obstacle_configuration: str,
                 inflation_radius: float,
                 initial_base_rng_x=(-1.0, 1.0),
                 initial_base_rng_y=(-1.0, 1.0),
                 initial_base_rng_yaw=(0, 0)):
        super(ObstacleConfigMap, self).__init__(world_type=world_type,
                                                initial_base_rng_x=initial_base_rng_x,
                                                initial_base_rng_y=initial_base_rng_y,
                                                initial_base_rng_yaw=initial_base_rng_yaw,
                                                robot_frame_id=robot_frame_id,
                                                requires_spawn=False,
                                                inflation_radius=inflation_radius)
        assert obstacle_configuration in ['none', 'inpath'], obstacle_configuration
        self.obstacle_configuration = obstacle_configuration

    @staticmethod
    def _add_inpath_obstacles() -> List[SpawnObject]:
        positions = []
        # between robot and pick-table
        positions.append((1.5, 0))
        # between robot and door-kallax.
        positions.append((-0.0, 1.5))
        # between robot and drawer-kallax
        positions.append((-1.5, -0.0))

        spawn_objects = []
        for i, pos in enumerate(positions):
            obstacle_pose = Pose(Point(pos[0], pos[1], 0.24), Quaternion(0, 0, 0, 1))
            obj = SpawnObject(f"obstacle{i}", WorldObjects.kallax2, obstacle_pose, "world")
            spawn_objects.append((obj))
        return spawn_objects

    def get_varying_scene_objects(self) -> List[SpawnObject]:
        if self.obstacle_configuration == 'inpath':
            return self._add_inpath_obstacles()
        elif self.obstacle_configuration == 'none':
            return []
        else:
            raise NotImplementedError(self.obstacle_configuration)


class PickNPlaceChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "picknplace"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: MobileManipulationEnv, default_head_start: float, obstacle_configuration: str):
        map = ObstacleConfigMap(world_type=env.get_world(),
                                obstacle_configuration=obstacle_configuration,
                                initial_base_rng_yaw=(-0.5 * np.pi, 0.5 * np.pi),
                                robot_frame_id=env.robot_config["frame_id"],
                                inflation_radius=env.get_inflation_radius())
        super(PickNPlaceChainedTask, self).__init__(env=env,
                                                    map=map,
                                                    close_gripper_at_start=False,
                                                    default_head_start=default_head_start)
        self._pick_obj = WorldObjects.muesli2
        self._pick_table = WorldObjects.reemc_table_low
        self._place_table = WorldObjects.reemc_table_low
        self._ee_fn = LinearPlannerWrapper

    def get_goal_objects(self, start_table_pos=Point(x=3.3, y=0, z=0), end_table_rng=(-1.5, 2, -3, -2.5)):
        objects = []
        # pick table
        start_table_pose = Pose(start_table_pos, Quaternion(0, 0, 1, 0))
        objects.append(SpawnObject("pick_table", self._pick_table, start_table_pose, "world"))
        # place table
        endx = random.uniform(end_table_rng[0], end_table_rng[1])
        endy = random.uniform(end_table_rng[2], end_table_rng[3])
        end_table_pose = Pose(Point(x=endx, y=endy, z=0), Quaternion(0, 0, 1, 1))
        objects.append(SpawnObject("place_table", self._place_table, end_table_pose, "world"))

        # place object on edge of the table (relative to the table position)
        # NOTE: won't be correct yet if table not in front of robot
        pose_on_table = Pose(Point(x=self._pick_table.x / 2 - 0.1,
                                   y=random.uniform(-self._pick_table.y + 0.1, self._pick_table.y - 0.1) / 2,
                                   z=self._pick_table.z + self._pick_obj.z + 0.01),
                             Quaternion(0, 0, 1, 1))
        self.pick_obj_pose = list_to_pose(
            multiply_tfs(pose_to_list(start_table_pose), pose_to_list(pose_on_table), False))
        objects.append(SpawnObject("pick_obj", self._pick_obj, self.pick_obj_pose, "world"))

        # target position to place the object
        self.place_obj_pose = list_to_pose(
            multiply_tfs(pose_to_list(end_table_pose), pose_to_list(pose_on_table), False))
        return objects

    def draw_goal(self) -> List[TaskGoal]:
        # NOTE: 'in_front' goals assume current rotation of tables relative to robot
        # pick goals
        world_target_pos = self.pick_obj_pose.position
        obj_loc = [world_target_pos.x, world_target_pos.y, world_target_pos.z - 0.04] + [0, 0, 0]
        in_front_of_obj_loc = copy.deepcopy(obj_loc)
        in_front_of_obj_loc[0] -= 0.2

        # place goals
        world_end_target_pos = self.place_obj_pose.position
        place_loc = [world_end_target_pos.x, world_end_target_pos.y + 0.05, world_end_target_pos.z + 0.05] + [0, 0,
                                                                                                              -np.pi / 2]
        in_front_of_place_loc = copy.deepcopy(place_loc)
        in_front_of_place_loc[1] += 0.2

        place_loc[2] -= 0.05

        return [TaskGoal(gripper_goal_tip=in_front_of_obj_loc, end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=self._default_head_start, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=obj_loc, end_action=GripperActions.GRASP,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=in_front_of_place_loc, end_action=GripperActions.NONE,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=place_loc, end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0, ee_fn=self._ee_fn)]


class DoorChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "door"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: MobileManipulationEnv, default_head_start: float, obstacle_configuration: str):
        map = ObstacleConfigMap(world_type=env.get_world(),
                                obstacle_configuration=obstacle_configuration,
                                initial_base_rng_yaw=(0.0 * np.pi, 1.0 * np.pi),
                                robot_frame_id=env.robot_config["frame_id"],
                                inflation_radius=env.get_inflation_radius())
        super(DoorChainedTask, self).__init__(env=env,
                                              map=map,
                                              close_gripper_at_start=False,
                                              default_head_start=default_head_start)
        self._shelf = WorldObjects.kallax2
        self.kallax2_origin_to_door_pose = list_to_pose([0.02, -0.17, -0.017, 0, 0, -1, 1])

    def get_goal_objects(self, shelf_pos=Point(x=0.0, y=3.3, z=0.24)):
        objects = []
        objects.append(SpawnObject("Kallax2_bottom", self._shelf, Pose(shelf_pos, Quaternion(0, 0, 0, 1)), "world"))
        p = copy.deepcopy(shelf_pos)
        p.z = 0.65
        self.target_shelf_pose = Pose(p, Quaternion(0, 0, 0, 1))
        objects.append(SpawnObject("target_shelf", self._shelf, self.target_shelf_pose, "world"))
        return objects

    @staticmethod
    def gmm_obj_origin_to_tip(gmm_model_path, obj_origin):
        """creating a dummy planner is simpler than to write another model.csv parser in python"""
        identity_tf = [0, 0, 0, 0, 0, 0, 1]
        assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"
        planner = GMMPlanner(identity_tf, identity_tf, identity_tf, identity_tf, 0., 0., 0., 0., 1., 0., 0., True,
                             identity_tf[:3], identity_tf[3:], str(gmm_model_path), 0.0)
        return planner.obj_origin_to_tip(obj_origin)

    def draw_goal(self) -> List[TaskGoal]:
        # grasp goal
        door_pose_closed = list_to_pose(
            multiply_tfs(pose_to_list(self.target_shelf_pose), pose_to_list(self.kallax2_origin_to_door_pose), False))
        obj_origin_goal = copy.deepcopy(door_pose_closed)
        obj_origin_goal.position.y += 0.01
        obj_origin_goal = pose_to_list(obj_origin_goal)

        motion_plan_grasp = str(self._motion_model_path / "GMM_grasp_KallaxTuer.csv")
        grasp_goal = TaskGoal(
            gripper_goal_tip=DoorChainedTask.gmm_obj_origin_to_tip(motion_plan_grasp, obj_origin_goal),
            end_action=GripperActions.GRASP,
            success_thres_dist=self._success_thres_dist,
            success_thres_rot=self._success_thres_rot,
            head_start=self._default_head_start,
            ee_fn=partial(GMMPlannerWrapper, gmm_model_path=motion_plan_grasp, robot_config=self.env.robot_config))

        # opening goal: expects the object pose in the beginning of the movement
        motion_plan_opening = str(self._motion_model_path / "GMM_move_KallaxTuer.csv")
        opening_goal = TaskGoal(
            gripper_goal_tip=DoorChainedTask.gmm_obj_origin_to_tip(motion_plan_opening, obj_origin_goal),
            end_action=GripperActions.OPEN,
            success_thres_dist=self._success_thres_dist,
            success_thres_rot=self._success_thres_rot,
            head_start=self.SUBGOAL_PAUSE,
            ee_fn=partial(GMMPlannerWrapper, gmm_model_path=motion_plan_opening, robot_config=self.env.robot_config))
        return [grasp_goal, opening_goal]


class DrawerChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "drawer"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: MobileManipulationEnv, default_head_start: float, obstacle_configuration: str):
        map = ObstacleConfigMap(world_type=env.get_world(),
                                obstacle_configuration=obstacle_configuration,
                                initial_base_rng_yaw=(0.5 * np.pi, 1.5 * np.pi),
                                robot_frame_id=env.robot_config["frame_id"],
                                inflation_radius=env.get_inflation_radius())
        super(DrawerChainedTask, self).__init__(env=env,
                                                map=map,
                                                close_gripper_at_start=False,
                                                default_head_start=default_head_start)
        self._drawer = WorldObjects.kallax
        self.kallax_origin_to_drawer_pose = list_to_pose([-0., 0.03, -0.185, 0, 0, 0, 1])

    def get_goal_objects(self, drawer_pos=Point(x=-3.3, y=0.0, z=0.24)) -> List[SpawnObject]:
        objects = []
        objects.append(SpawnObject("Kallax_bottom", self._drawer, Pose(drawer_pos, Quaternion(0, 0, 1, 1)), "world"))

        p = copy.deepcopy(drawer_pos)
        p.z = 0.65
        self.target_drawer_pose = Pose(p, Quaternion(0, 0, 1, 1))
        objects.append(SpawnObject("target_drawer", self._drawer, self.target_drawer_pose, "world"))
        time.sleep(1)
        return objects

    def draw_goal(self) -> List[TaskGoal]:
        # grasp goal
        door_pose_closed = list_to_pose(
            multiply_tfs(pose_to_list(self.target_drawer_pose), pose_to_list(self.kallax_origin_to_drawer_pose), False))
        obj_origin_goal = [door_pose_closed.position.x + 0.04, door_pose_closed.position.y,
                           door_pose_closed.position.z + 0.05,
                           0, 0, 0, 1]
        motion_plan_grasp = str(self._motion_model_path / "GMM_grasp_KallaxDrawer.csv")
        grasp_goal = TaskGoal(
            gripper_goal_tip=DoorChainedTask.gmm_obj_origin_to_tip(motion_plan_grasp, obj_origin_goal),
            end_action=GripperActions.GRASP,
            success_thres_dist=self._success_thres_dist,
            success_thres_rot=self._success_thres_rot,
            head_start=self._default_head_start,
            ee_fn=partial(GMMPlannerWrapper, gmm_model_path=motion_plan_grasp, robot_config=self.env.robot_config))

        motion_plan_opening = str(self._motion_model_path / "GMM_move_KallaxDrawer.csv")
        opening_goal = TaskGoal(
            gripper_goal_tip=DoorChainedTask.gmm_obj_origin_to_tip(motion_plan_opening, obj_origin_goal),
            end_action=GripperActions.OPEN,
            success_thres_dist=self._success_thres_dist,
            success_thres_rot=self._success_thres_rot,
            head_start=self.SUBGOAL_PAUSE,
            ee_fn=partial(GMMPlannerWrapper, gmm_model_path=motion_plan_opening, robot_config=self.env.robot_config))
        return [grasp_goal, opening_goal]
