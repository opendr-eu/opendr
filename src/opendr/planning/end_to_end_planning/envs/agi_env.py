# Copyright 2020-2022 OpenDR European Project
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
import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from nav_msgs.msg import Path
from webots_ros.msg import BoolStamped


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.atan2(t3, t4)

    return yaw_z / np.pi * 180  # in radians


class AgiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AgiEnv, self).__init__()

        # Gym elements
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = spaces.Dict(
            {'depth_cam': spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
             'moving_target': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)})

        self.action_dictionary = {0: (np.cos(22.5 / 180 * np.pi), np.sin(22.5 / 180 * np.pi), 1),
                                  1: (np.cos(22.5 / 180 * np.pi), np.sin(22.5 / 180 * np.pi), 0),
                                  2: (1, 0, 1),
                                  3: (1, 0, 0),
                                  4: (1, 0, -1),
                                  5: (np.cos(22.5 / 180 * np.pi), -np.sin(22.5 / 180 * np.pi), 0),
                                  6: (np.cos(22.5 / 180 * np.pi), -np.sin(22.5 / 180 * np.pi), -1),
                                  7: (0, 0, 2),
                                  8: (0, 0, -2)}
        self.step_length = 1  # meter

        self.current_position = PoseStamped().pose.position
        self.current_yaw = 0
        self.range_image = np.ones((64, 64), dtype=np.float32)
        self.collision_flag = False
        self.model_name = ""

        # ROS connection
        rospy.init_node('agi_gym_environment')
        self.r = rospy.Rate(10)
        self.ros_pub_pose = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.ros_pub_target = rospy.Publisher('target_position', PoseStamped, queue_size=10)
        self.ros_pub_trajectory = rospy.Publisher('uav_trajectory', Path, queue_size=10)
        self.ros_pub_global_trajectory = rospy.Publisher('uav_global_trajectory', Path, queue_size=10)
        self.global_traj = Path()
        self.uav_trajectory = Path()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/range_image_raw", Float32MultiArray, self.range_image_callback)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        self.r.sleep()
        rospy.Subscriber("/" + self.model_name + "/touch_sensor/value", BoolStamped, self.collision_callback)

        self.target_y = -22
        self.target_y_list = [-22, -16, -10, -4, 2, 7, 12]  # evaluation map:[-22, -16, -10, -4, 2, 8, 14, 20, 26, 32]
        self.target_z = 2.5
        self.start_x = -10
        self.forward_direction = True
        self.parkour_length = 30
        self.episode_counter = 0

        self.set_target()
        self.r.sleep()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) - vo[1] * np.sin(
            self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                                                self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[2]])
        self.observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.copy(self.vector_observation)}
        self.r.sleep()

        self.image_count = 0

    def step(self, discrete_action):
        if self.current_position == PoseStamped().pose.position:
            rospy.loginfo("Gym environment is not reading mavros position")
            return self.observation_space.sample(), np.random.random(1), False, {}
        action = self.action_dictionary[discrete_action]
        action = (action[0] * self.step_length, action[1] * self.step_length, action[2])
        prev_x = self.current_position.x
        if self.forward_direction:
            self.go_position(
                self.current_position.x + action[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) - action[
                    1] * np.sin(self.current_yaw * 22.5 / 180 * np.pi),
                self.current_position.y + action[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + action[
                    1] * np.cos(self.current_yaw * 22.5 / 180 * np.pi), self.target_z, yaw=self.current_yaw + action[2],
                check_collision=True)
        else:
            self.go_position(
                self.current_position.x - action[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + action[
                    1] * np.sin(self.current_yaw * 22.5 / 180 * np.pi),
                self.current_position.y - action[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) - action[
                    1] * np.cos(self.current_yaw * 22.5 / 180 * np.pi), self.target_z, yaw=self.current_yaw + action[2],
                check_collision=True)
        self.update_trajectory()

        dx = np.abs(self.current_position.x - prev_x)
        dy = np.abs(self.current_position.y - self.target_position.y)
        dyaw = np.abs(self.current_yaw)
        reward = 2 * dx - 0.4 * dy - 0.3 * dyaw

        # set new observation
        if self.forward_direction:
            self.set_target()
            vo = self.difference_between_points(self.target_position, self.current_position)
            self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
                self.current_yaw * 22.5 / 180 * np.pi),
                                                -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                                                    self.current_yaw * 22.5 / 180 * np.pi),
                                                vo[2]])
            self.observation = {'depth_cam': np.copy(self.range_image),
                                'moving_target': np.copy(self.vector_observation)}
            finish_passed = (self.current_position.x > self.parkour_length + self.start_x)
        else:
            self.set_target()
            vo = self.difference_between_points(self.current_position, self.target_position)
            self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
                self.current_yaw * 22.5 / 180 * np.pi),
                                                -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                                                    self.current_yaw * 22.5 / 180 * np.pi),
                                                vo[2]])
            self.observation = {'depth_cam': np.copy(self.range_image),
                                'moving_target': np.copy(self.vector_observation)}
            finish_passed = (self.current_position.x < self.start_x - self.parkour_length)

        # check done
        if finish_passed:
            reward = 20
            done = True
        elif abs(self.current_position.y - self.target_y) > 5:
            reward = -10
            done = True
        elif self.collision_flag:
            reward = -20
            done = True
        else:
            done = False

        info = {"current_position": self.current_position, "finish_passed": finish_passed}
        return self.observation, reward, done, info

    def reset(self):
        if self.current_position == PoseStamped().pose.position:
            rospy.loginfo("Gym environment is not reading mavros position")
            return self.observation_space.sample()
        self.target_y = np.random.choice(self.target_y_list)
        self.go_position(self.current_position.x, self.current_position.y, 8)
        self.go_position(self.start_x, self.current_position.y, 8)
        self.go_position(self.start_x, self.target_y, self.target_z)
        self.uav_trajectory.header.frame_id = "map"
        self.update_trajectory()
        self.publish_global_trajectory()

        self.collision_flag = False
        self.set_target()
        if self.forward_direction:
            self.vector_observation = self.difference_between_points(self.target_position, self.current_position)
        else:
            self.vector_observation = self.difference_between_points(self.current_position, self.target_position)
        self.observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.copy(self.vector_observation)}
        return self.observation

    def set_target(self):
        self.target_position = PoseStamped().pose.position
        if self.forward_direction:
            self.target_position.x = self.current_position.x + 5
        else:
            self.target_position.x = self.current_position.x - 5
        self.target_position.y = self.target_y
        self.target_position.z = self.target_z
        self.publish_target()

    def render(self, mode='human', close=False):
        pass

    def pose_callback(self, data):
        self.current_position = data.pose.position

    def range_image_callback(self, data):
        self.range_image = ((np.clip(np.array(data.data).reshape((64, 64, 1)), 0, 15) / 15.)*255).astype(np.uint8)

    def model_name_callback(self, data):
        if data.data[:5] == "robot":
            self.model_name = data.data

    def collision_callback(self, data):
        if data.data:
            self.collision_flag = True
            # print("colliiiddeeee")

    def go_position(self, x, y, z, yaw=0, check_collision=False):
        if yaw > 4:
            yaw = 4
        if yaw < -4:
            yaw = -4
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        # goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        quat_z_yaw_dict = {-4: -0.7071068, -3: -0.5555702, -2: -0.3826834, -1: -0.1950903, 0: 0.0, 1: 0.1950903,
                           2: 0.3826834, 3: 0.5555702, 4: 0.7071068}
        quat_w_yaw_dict = {-4: 0.7071068, -3: 0.8314696, -2: 0.9238795, -1: 0.9807853, 0: 1.0, 1: 0.9807853,
                           2: 0.9238795, 3: 0.8314696, 4: 0.7071068}
        if self.forward_direction:
            goal.pose.orientation.z = quat_z_yaw_dict[yaw]
            goal.pose.orientation.w = quat_w_yaw_dict[yaw]
        else:
            goal.pose.orientation.z = -quat_w_yaw_dict[yaw]
            goal.pose.orientation.w = quat_z_yaw_dict[yaw]
        self.current_yaw = yaw
        self.ros_pub_pose.publish(goal)
        self.r.sleep()
        while self.distance_between_points(goal.pose.position, self.current_position) > 0.1:
            if check_collision and self.collision_flag:
                return
            self.ros_pub_pose.publish(goal)
            self.r.sleep()

    def publish_target(self):
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position = self.target_position

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0
        self.ros_pub_target.publish(goal)

    def update_trajectory(self):
        new_point = PoseStamped()
        new_point.header.seq = 1
        new_point.header.stamp = rospy.Time.now()
        new_point.header.frame_id = "map"
        new_point.pose.position.x = self.current_position.x
        new_point.pose.position.y = self.current_position.y
        new_point.pose.position.z = self.current_position.z
        self.uav_trajectory.poses.append(new_point)
        self.ros_pub_trajectory.publish(self.uav_trajectory)

    def publish_global_trajectory(self):
        self.global_traj.header.frame_id = "map"
        new_point = PoseStamped()
        new_point.header.seq = 1
        new_point.header.stamp = rospy.Time.now()
        new_point.header.frame_id = "map"
        new_point.pose.position.x = self.start_x
        new_point.pose.position.y = self.target_y
        new_point.pose.position.z = self.target_z
        self.global_traj.poses.append(new_point)
        new_point = PoseStamped()
        new_point.header.seq = 1
        new_point.header.stamp = rospy.Time.now()
        new_point.header.frame_id = "map"
        if self.forward_direction:
            new_point.pose.position.x = self.start_x + self.parkour_length
        else:
            new_point.pose.position.x = self.start_x - self.parkour_length
        new_point.pose.position.y = self.target_y
        new_point.pose.position.z = self.target_z
        self.global_traj.poses.append(new_point)
        self.ros_pub_global_trajectory.publish(self.global_traj)

    def distance_between_points(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return np.sqrt(x * x + y * y + z * z)

    def difference_between_points(self, p1, p2):
        return np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
