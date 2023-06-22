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
import gym
from cv_bridge import CvBridge
import webots_ros.srv
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from nav_msgs.msg import Path
from webots_ros.msg import BoolStamped
from sensor_msgs.msg import Imu, Image
from opendr.planning.end_to_end_planning.utils.obstacle_randomizer import ObstacleRandomizer
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_from_quaternion
from opendr.planning.end_to_end_planning.utils.euler_quaternion_transformations import euler_to_quaternion


class UAVDepthPlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, no_dynamics=True, discrete_actions=False):
        super(UAVDepthPlanningEnv, self).__init__()

        # Gym elements
        self.observation_space = spaces.Dict(
            {'depth_cam': spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
             'moving_target': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)})
        self.is_discrete_actions = discrete_actions
        if self.is_discrete_actions:
            self.action_space = gym.spaces.Discrete(7)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float64)
        self.action_dictionary = {0: (1, 1),  # used for discrete actions
                                  1: (1, 0),
                                  2: (0, 1),
                                  3: (0, 0),
                                  4: (0, -1),
                                  5: (-1, 0),
                                  6: (-1, -1)}
        self.step_length = 1  # meter

        self.current_position = PoseStamped().pose.position
        self.current_yaw = 0
        self.current_orientation = PoseStamped().pose.orientation
        self.range_image = np.ones((64, 64, 1), dtype=np.float32)
        self.collision_flag = False
        self.safety1_flag = False
        self.safety2_flag = False
        self.enable_safety_reward = True
        self.model_name = ""
        self.target_y = 0
        self.target_z = 2.5
        self.start_x = -10
        self.forward_direction = True
        self.parkour_length = 30
        self.episode_counter = 0
        self.closer_object_length = 5
        self.no_dynamics = no_dynamics

        # ROS connection
        self.bridge = CvBridge()
        rospy.init_node('gym_depth_planning_environment')
        self.r = rospy.Rate(25)
        self.ros_pub_pose = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.ros_pub_target = rospy.Publisher('target_position', PoseStamped, queue_size=10)
        self.ros_pub_trajectory = rospy.Publisher('uav_trajectory', Path, queue_size=10)
        self.ros_pub_global_trajectory = rospy.Publisher('uav_global_trajectory', Path, queue_size=10)
        self.global_traj = Path()
        self.uav_trajectory = Path()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        counter = 0
        rospy.loginfo("Waiting for webots model to start!")
        while self.model_name == "":
            self.r.sleep()
            counter += 1
            if counter > 25:
                break
        if self.model_name == "":
            rospy.loginfo("Webots model is not started!")
            return
        self.randomizer = ObstacleRandomizer(self.model_name)
        rospy.Subscriber("/touch_sensor_collision/value", BoolStamped, self.collision_callback)
        rospy.Subscriber("/touch_sensor_safety1/value", BoolStamped, self.safety1_callback)
        rospy.Subscriber("/touch_sensor_safety2/value", BoolStamped, self.safety2_callback)
        rospy.Subscriber("/range_finder/range_image", Image, self.range_callback, queue_size=1)
        self.ros_srv_touch_sensor_collision_enable = rospy.ServiceProxy(
            "/touch_sensor_collision/enable", webots_ros.srv.set_int)
        self.ros_srv_touch_sensor_safety1_enable = rospy.ServiceProxy(
            "/touch_sensor_safety1/enable", webots_ros.srv.set_int)
        self.ros_srv_touch_sensor_safety2_enable = rospy.ServiceProxy(
            "/touch_sensor_safety2/enable", webots_ros.srv.set_int)
        self.ros_srv_range_sensor_enable = rospy.ServiceProxy(
            "/range_finder/enable", webots_ros.srv.set_int)
        try:
            self.ros_srv_touch_sensor_collision_enable(1)
            self.ros_srv_range_sensor_enable(1)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            self.ros_srv_touch_sensor_safety1_enable(1)
            self.ros_srv_touch_sensor_safety2_enable(1)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

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

        if no_dynamics:
            self.ros_srv_gps_sensor_enable = rospy.ServiceProxy(
                "/gps/enable", webots_ros.srv.set_int)
            self.ros_srv_inertial_unit_enable = rospy.ServiceProxy(
                "/inertial_unit/enable", webots_ros.srv.set_int)
            self.ros_srv_get_self = rospy.ServiceProxy(
                "/supervisor/get_self", webots_ros.srv.get_uint64)
            self.ros_srv_get_field = rospy.ServiceProxy(
                "/supervisor/node/get_field", webots_ros.srv.node_get_field)
            self.ros_srv_field_set_v3 = rospy.ServiceProxy(
                "/supervisor/field/set_vec3f", webots_ros.srv.field_set_vec3f)
            self.ros_srv_field_set_rotation = rospy.ServiceProxy(
                "/supervisor/field/set_rotation", webots_ros.srv.field_set_rotation)
            rospy.Subscriber("/inertial_unit/quaternion", Imu, self.imu_callback)
            rospy.Subscriber("/gps/values", PointStamped, self.gps_callback)
            try:
                self.ros_srv_gps_sensor_enable(1)
                self.ros_srv_inertial_unit_enable(1)
            except rospy.ServiceException as exc:
                print("Service did not process request: " + str(exc))
            try:
                resp1 = self.ros_srv_get_self(True)
                self.robot_node_id = resp1.value
            except rospy.ServiceException as exc:
                print("Service did not process request: " + str(exc))
            try:
                resp1 = self.ros_srv_get_field(self.robot_node_id, 'translation', False)
                resp2 = self.ros_srv_get_field(self.robot_node_id, 'rotation', False)
                self.robot_translation_field = resp1.field
                self.robot_rotation_field = resp2.field
            except rospy.ServiceException as exc:
                print("Service did not process request: " + str(exc))

    def step(self, action):
        if self.model_name == "":
            rospy.loginfo("Gym environment cannot connect to Webots")
            return self.observation_space.sample(), np.random.random(1), False, {}
        if self.is_discrete_actions:
            action = self.action_dictionary[action]
        forward_step = np.cos(action[0] * 22.5 / 180 * np.pi)
        side_step = np.sin(action[0] * 22.5 / 180 * np.pi)
        yaw_step = action[1] * 22.5 / 180 * np.pi

        # take the step
        prev_x = self.current_position.x
        if self.forward_direction:
            yaw = self.current_yaw + yaw_step
            if yaw > np.pi / 2:
                yaw = np.pi / 2
            if yaw < -np.pi / 2:
                yaw = -np.pi / 2
            self.go_position(
                self.current_position.x + forward_step * np.cos(self.current_yaw) - side_step * np.sin(
                    self.current_yaw),
                self.current_position.y + forward_step * np.sin(self.current_yaw) + side_step * np.cos(
                    self.current_yaw),
                self.target_z, yaw=yaw,
                check_collision=True)
        else:
            yaw = self.current_yaw + yaw_step
            if np.abs(yaw) < np.pi / 2:
                yaw = np.sign(yaw) * np.pi / 2
            self.go_position(
                self.current_position.x + forward_step * np.cos(self.current_yaw) - side_step * np.sin(
                    self.current_yaw),
                self.current_position.y + forward_step * np.sin(self.current_yaw) + side_step * np.cos(
                    self.current_yaw),
                self.target_z, yaw=yaw,
                check_collision=True)
        self.update_trajectory()

        self.r.sleep()

        dx = np.abs(self.current_position.x - prev_x)
        dy = np.abs(self.current_position.y - self.target_position.y)
        dyaw = np.abs(self.current_yaw)

        # calculate reward
        reward = 2 * dx - 0.4 * dy - 0.3 * dyaw
        if self.enable_safety_reward:
            if self.safety2_flag:
                reward -= 2
            if self.safety1_flag:
                reward -= 10

        # set new observation
        if self.forward_direction:
            self.set_target()
            vo = self.difference_between_points(self.target_position, self.current_position)
            self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw) + vo[1] * np.sin(self.current_yaw),
                                                -vo[0] * np.sin(self.current_yaw) + vo[1] * np.cos(self.current_yaw),
                                                vo[2]])
            self.observation = {'depth_cam': np.copy(self.range_image),
                                'moving_target': np.copy(self.vector_observation)}
            finish_passed = (self.current_position.x > self.parkour_length + self.start_x)
        else:
            self.set_target()
            vo = self.difference_between_points(self.current_position, self.target_position)
            self.vector_observation = np.array(
                [vo[0] * np.cos(self.current_yaw + np.pi) + vo[1] * np.sin(self.current_yaw + np.pi),
                 -vo[0] * np.sin(self.current_yaw + np.pi) + vo[1] * np.cos(self.current_yaw + np.pi),
                 vo[2]])
            self.observation = {'depth_cam': np.copy(self.range_image),
                                'moving_target': np.copy(self.vector_observation)}
            finish_passed = (self.current_position.x < self.start_x - self.parkour_length)

        # check done and update reward
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

        info = {"current_position": self.current_position, "finish_passed": finish_passed,
                "safety_flags": [self.safety1_flag, self.safety2_flag], "closer_object": self.closer_object_length}
        self.safety1_flag = False
        self.safety2_flag = False
        return self.observation, reward, done, info

    def reset(self):
        if self.model_name == "":
            rospy.loginfo("Gym environment cannot connect to Webots")
            return self.observation_space.sample()
        if self.no_dynamics:
            self.go_position(self.start_x, self.target_y + np.random.uniform(-0.5, 0.5), self.target_z,
                             yaw=(1 - self.forward_direction) * np.pi)
        else:
            self.go_position(self.current_position.x, self.current_position.y, 8)
            self.go_position(self.start_x, self.current_position.y, 8)
            self.go_position(self.start_x, self.target_y, 8)
            self.go_position(self.start_x, self.target_y + np.random.uniform(-0.5, 0.5), self.target_z)
        self.r.sleep()
        self.uav_trajectory.header.frame_id = "map"
        self.update_trajectory()
        self.publish_global_trajectory()

        self.collision_flag = False
        self.safety1_flag = False
        self.safety2_flag = False
        self.set_target()
        if self.forward_direction:
            self.vector_observation = self.difference_between_points(self.target_position, self.current_position)
        else:
            self.vector_observation = self.difference_between_points(self.current_position, self.target_position)
        self.observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.copy(self.vector_observation)}
        self.randomizer.randomize_environment()
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
        self.current_orientation = data.pose.orientation
        self.current_yaw = euler_from_quaternion(data.pose.orientation)["yaw"]

    def range_callback(self, data):
        image_arr = self.bridge.imgmsg_to_cv2(data)
        self.range_image = ((np.clip(image_arr.reshape((64, 64, 1)), 0, 15) / 15.) * 255).astype(np.uint8)

    def model_name_callback(self, data):
        if data.data[:5] == "robot":
            self.model_name = data.data
        if data.data[:4] == "quad":
            self.model_name = data.data

    def collision_callback(self, data):
        if data.data:
            self.collision_flag = True

    def safety1_callback(self, data):
        if data.data:
            self.safety1_flag = True

    def safety2_callback(self, data):
        if data.data:
            self.safety2_flag = True

    def gps_callback(self, data):  # for no dynamics
        self.current_position.x = -data.point.x
        self.current_position.y = -data.point.y
        self.current_position.z = data.point.z

    def imu_callback(self, data):  # for no dynamics
        self.current_orientation = data.orientation
        self.current_yaw = euler_from_quaternion(data.orientation)["yaw"]

    def go_position(self, x, y, z, yaw=0, check_collision=False):
        if self.no_dynamics:
            goal = PoseStamped()

            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()

            goal.pose.position.x = -x
            goal.pose.position.y = -y
            goal.pose.position.z = z

            goal.pose.orientation = euler_to_quaternion(np.pi/2, 0, -np.pi/2+yaw)
            try:
                self.ros_srv_field_set_v3(self.robot_translation_field, 0, goal.pose.position)
                self.ros_srv_field_set_rotation(self.robot_rotation_field, 0, goal.pose.orientation)
                self.r.sleep()
            except rospy.ServiceException as exc:
                print("Service did not process request: " + str(exc))
        else:
            goal = PoseStamped()

            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()

            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = z

            goal.pose.orientation = euler_to_quaternion(0, 0, yaw)
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
