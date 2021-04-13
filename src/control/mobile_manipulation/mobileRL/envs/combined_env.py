import pickle

import numpy as np
import torch
from PIL import Image, ImageDraw
from gym import spaces, Env
from matplotlib import pyplot as plt
from pybindings import RobotObs
from torchvision import transforms

from control.mobile_manipulation.mobileRL.envs.eeplanner import EEPlanner
from control.mobile_manipulation.mobileRL.envs.env_utils import quaternion_to_yaw
from control.mobile_manipulation.mobileRL.envs.map import Map, DummyMap
from control.mobile_manipulation.mobileRL.envs.robotenv import RobotEnv, ActionRanges, unscale_action


class CombinedEnv(Env):
    metadata = {'render.modes': []}

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_robot']
        state['_robot'] = pickle.dumps(self._robot)
        return state

    def __setstate__(self, state):
        _robot = pickle.loads(state.pop('_robot'))
        self.__dict__ = state
        self._robot = _robot

    def __init__(self,
                 robot_env: RobotEnv,
                 ik_fail_thresh: int,
                 learn_vel_norm: bool,
                 slow_down_real_exec: float,
                 use_map_obs: bool,
                 flatten_obs: bool = False):
        self._robot = robot_env
        self._ee_planner: EEPlanner = None
        # each task wrapper should set a map at initialisation if needed
        # placeholder with no computational costs for when not needed
        self.map = DummyMap()

        self._ik_fail_thresh = ik_fail_thresh
        self._learn_vel_norm = learn_vel_norm
        self._slow_down_real_execution = slow_down_real_exec

        self._use_map_obs = use_map_obs
        # Only so we can easily keep compatibility with stable-baselines which needs flat observations
        self._flatten_obs = flatten_obs
        self.robot_state_dim = self._robot.get_obs_dim()
        print(f"Detected robot state dim: {self.robot_state_dim}")

        self.action_names, self._min_actions, self._max_actions = ActionRanges.get_ranges(env_name=self._robot.env_name,
                                                                                          strategy=self._robot._strategy,
                                                                                          learn_vel_norm=(learn_vel_norm != -1))
        print(f"Actions to learn: {self.action_names}")
        self.action_dim = len(self._min_actions)

        # NOTE: env does all the action scaling according to _min_actions and _max_actions
        # and just expects actions in the range [-1, 1] from the agent
        self.action_space = spaces.Box(low=np.array(self.action_dim * [-1.0]),
                                       high=np.array(self.action_dim * [1.0]),
                                       shape=[self.action_dim])
        self.reward_range = (-10, 0)

    def get_inflation_radius(self) -> float:
        return max(self._robot.robot_base_size) / 2

    def local_map_in_collision(self, local_map: np.ndarray) -> bool:
        # TODO: can we account for the arm as well? More complex base shapes?
        robot_base_size_pixels = (torch.Tensor(self._robot.robot_base_size) / self.map._resolution).ceil().to(torch.int)
        crop_fn = transforms.CenterCrop(size=robot_base_size_pixels)
        center = crop_fn(torch.Tensor(local_map))
        return bool(center.sum() > 0)

    @property
    def observation_space(self):
        if self._flatten_obs:
            n = self.robot_state_dim
            if self._use_map_obs:
                n += np.prod(self.map.output_size)
            return spaces.Box(low=-100, high=100, shape=[n])
        else:
            s = spaces.Box(low=-100, high=100, shape=[self.robot_state_dim])
            if self._use_map_obs:
                s = spaces.Tuple([s,
                                  # map atm returns a 2D image w/o any channel
                                  spaces.Box(low=-0.1, high=1.1, shape=list(self.map.output_size) + [1],
                                             dtype=np.float)])
            return s

    def set_map(self, map: Map):
        self.map = map

    def set_ee_planner(self,
                       ee_planner: EEPlanner,
                       success_thres_dist: float,
                       success_thres_rot: float):
        is_analytical = self._robot.is_analytical_world()
        robot_obs = self._robot.get_robot_obs()

        self._ee_planner = ee_planner
        ee_obs_train_freq = self._ee_planner.reset(success_thres_dist=success_thres_dist,
                                                   success_thres_rot=success_thres_rot,
                                                   robot_obs=robot_obs,
                                                   slow_down_factor=self._slow_down_real_execution if not is_analytical else 1,
                                                   is_analytic_env=is_analytical)
        if self._robot.vis_env:
            self._robot.publish_marker(ee_planner.gripper_goal_wrist, 9999, "gripper_goal", "blue", 1.0)

        if self._robot.env_name == "hsr":
            self._robot.set_gripper_goal_wrist(ee_planner.gripper_goal_wrist)
        # returns a full obs, not just the new ee_obs to avoid having to stick it together correctly in other places
        return self._to_agent_obs(robot_obs, ee_obs_train_freq.ee_velocities_rel)

    def reset(self,
              initial_joint_distribution: str,
              success_thres_dist: float,
              success_thres_rot: float,
              ee_planner: EEPlanner,
              close_gripper: bool = False):
        # first clear map so robot cannot crash into objects when resetting, then respawn the scene
        self.map.clear()
        robot_obs = self._robot.reset(initial_base_pose=self.map.draw_initial_base_pose(ee_planner.gripper_goal_wrist if ee_planner is not None else None),
                                      initial_joint_distribution=initial_joint_distribution,
                                      success_thres_dist=success_thres_dist,
                                      success_thres_rot=success_thres_rot,
                                      close_gripper=close_gripper)
        self.nr_base_collisions = 0

        self.map.map_reset()
        if self.vis_env:
            self.map.publish_floorplan_rviz()
        # sometimes desirable to call set_ee_planner manually after the map has reset to first deduct goals from the map
        if ee_planner is not None:
            return self.set_ee_planner(ee_planner=ee_planner,
                                       success_thres_dist=success_thres_dist,
                                       success_thres_rot=success_thres_rot)

    def step(self, action):
        reward = 0.0
        vel_norm, base_actions = self._convert_policy_to_env_actions(action)

        ee_obs = self._ee_planner.step(self._robot.get_robot_obs(), vel_norm)
        robot_obs, robot_info = self._robot.step(base_actions, ee_obs.ee_velocities_world)
        ee_obs_train_freq = self._ee_planner.generate_obs_step(robot_obs)
        robot_info['ee_done'] = ee_obs_train_freq.done

        reward += robot_obs.reward + ee_obs_train_freq.reward
        if (self._learn_vel_norm != -1):
            reward -= self._learn_vel_norm * (self._robot.robot_config['base_vel_rng'] - vel_norm)

        local_map = self.map.get_local_map(robot_obs.base_tf, self._robot.get_world())
        robot_info['base_collision'] = self.local_map_in_collision(local_map)
        self.nr_base_collisions += robot_info['base_collision']
        robot_info['nr_base_collisions'] = self.nr_base_collisions

        # penalize stronger than ik failure
        reward -= 2 * robot_info['base_collision']

        done = ((robot_obs.done and ee_obs_train_freq.done)
                or (robot_info['nr_kin_failures'] >= self._ik_fail_thresh)
                or (robot_info['nr_base_collisions'] >= self._ik_fail_thresh))

        return self._to_agent_obs(robot_obs, ee_obs_train_freq.ee_velocities_rel, local_map), reward, done, robot_info

    def _to_agent_obs(self, robot_obs: RobotObs, ee_velocities_rel, local_map=None) -> list:
        obs_vector = (robot_obs.relative_gripper_tf
                      + ee_velocities_rel
                      + self._robot.world_to_relative_tf(self._ee_planner.gripper_goal_wrist)
                      + robot_obs.joint_values)
        if not self._use_map_obs:
            return obs_vector
        else:
            if local_map is None:
                local_map = self.map.get_local_map(robot_obs.base_tf, self._robot.get_world())
            if self._flatten_obs:
                obs_vector += local_map.flatten().tolist()
                return obs_vector
            else:
                local_map = local_map[:, :, np.newaxis]
                return obs_vector, local_map

    def _convert_policy_to_env_actions(self, actions):
        # NOTE: all actions are in range [-1, 1] at this point. Scale into target range either here or in the robot env.
        actions = list(actions)

        min_learned_vel_norm = 0.01

        if (self._learn_vel_norm != -1):
            # into range [min_vel_norm, robot_env.base_vel_rng]
            vel_norm = unscale_action(actions.pop(0), min_learned_vel_norm, self._robot.robot_config["base_vel_rng"])
        else:
            vel_norm = -1

        base_actions = actions
        return vel_norm, base_actions

    def plot_trajectory(self):
        f, ax = self.map.plot_floorplan()

        traj = np.array(self._robot.trajectory)
        if len(traj):
            base_traj, gripper_traj = traj[:, 0], traj[:, 1]
            gripper_traj[:, :2] = self.map.meter_to_pixels(gripper_traj[:, :2])
            base_traj[:, :2] = self.map.meter_to_pixels(base_traj[:, :2])

            ax.plot(gripper_traj[:, 0], gripper_traj[:, 1], ls=':', color='cyan', label=None)

            base_rot = quaternion_to_yaw(base_traj[:, 3:])
            ax.quiver(base_traj[:, 0], base_traj[:, 1], 1.0 * np.ones_like(base_rot), 1.0 * np.ones_like(base_rot),
                      angles=np.rad2deg(base_rot),
                      scale_units='dots', scale=0.09, width=0.002, headwidth=4, headlength=1.5, headaxislength=2,
                      pivot='middle',
                      color='orange', label=f'base')
        return f, ax

    def plot_local_map(self):
        robot_size_pixel = np.array(self._robot.robot_base_size) / self.map._resolution

        obs = self._robot.get_robot_obs()
        local_map = self.map.get_local_map(obs.base_tf, self._robot.get_world())
        img = Image.fromarray(local_map == 0).convert('1').convert('RGB')
        draw = ImageDraw.Draw(img)

        x = [img.size[0] / 2 - robot_size_pixel[0] / 2, img.size[0] / 2 + robot_size_pixel[0] / 2]
        y = [img.size[1] / 2 - robot_size_pixel[1] / 2, img.size[1] / 2 + robot_size_pixel[1] / 2]
        draw.rectangle(list(zip(x, y)), outline="green", width=1)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.asarray(img))
        return f, ax

    def plot_floorplan(self):
        obs = self._robot.get_robot_obs()
        return self.map.plot_floorplan(location=obs.base_tf)
