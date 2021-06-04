import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from webots_ros.msg import BoolStamped
import matplotlib.pyplot as plt
import tensorflow as tf
#from range_image_vae import RangeVAE


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    #t0 = +2.0 * (w * x + y * z)
    #t1 = +1.0 - 2.0 * (x * x + y * y)
    #roll_x = np.atan2(t0, t1)

    #t2 = +2.0 * (w * y - z * x)
    #t2 = +1.0 if t2 > +1.0 else t2
    #t2 = -1.0 if t2 < -1.0 else t2
    #pitch_y = np.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.atan2(t3, t4)

    return yaw_z/np.pi*180  # in radians

class AgiEnv_v2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AgiEnv_v2, self).__init__()

        # Gym elements
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64, 64, 2), dtype=np.float64)
        self.action_dictionary = {0: (np.cos(22.5/180*np.pi), np.sin(22.5/180*np.pi), 1),
                                  1: (np.cos(22.5/180*np.pi), np.sin(22.5/180*np.pi), 0),
                                  2: (1, 0, 1),
                                  3: (1, 0, 0),
                                  4: (1, 0, -1),
                                  5: (np.cos(22.5/180*np.pi), -np.sin(22.5/180*np.pi), 0),
                                  6: (np.cos(22.5/180*np.pi), -np.sin(22.5/180*np.pi), -1),
                                  7: (0, 0, 2),
                                  8: (0, 0, -2),}

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
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/range_image_raw", Float32MultiArray, self.range_image_callback)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        self.r.sleep()
        print(self.model_name)
        rospy.Subscriber("/" + self.model_name + "/touch_sensor/value", BoolStamped, self.collision_callback)

        self.target_y = -30 + np.random.randint(-1, 2)
        self.set_target()
        self.r.sleep()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) - vo[1] * np.sin(self.current_yaw * 22.5 / 180 * np.pi),
                                   vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(self.current_yaw * 22.5 / 180 * np.pi),
                                   vo[2]])
        self.observation = np.zeros((64, 64, 2))
        self.observation[:, :, 0] = np.copy(self.range_image)
        self.observation[0, 0:3, 1] = np.copy(self.vector_observation)
        #print(self.vector_observation)
        self.r.sleep()
        self.reset()

        self.image_count = 0

    def step(self, discrete_action):
        # take action
        action = self.action_dictionary[discrete_action]
        print(action)
        print("position before step", self.current_position)
        #plt.imshow(self.range_image, cmap='gray', vmin=0, vmax=1)
        #plt.show()
        self.go_position(self.current_position.x + action[0]*np.cos(self.current_yaw*22.5/180*np.pi) - action[1]*np.sin(self.current_yaw*22.5/180*np.pi),
                         self.current_position.y + action[0]*np.sin(self.current_yaw*22.5/180*np.pi) + action[1]*np.cos(self.current_yaw*22.5/180*np.pi), 5, yaw=self.current_yaw + action[2], check_collision=True)
        # calculate reward
        prev_distance = np.linalg.norm(self.vector_observation * np.array([1, 4, 1]))
        vo = self.difference_between_points(self.target_position, self.current_position)
        current_distance = np.linalg.norm(
            np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
                self.current_yaw * 22.5 / 180 * np.pi),
                      -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                          self.current_yaw * 22.5 / 180 * np.pi),
                      vo[2]]) * np.array([1, 4, 1]))
        delta_distance = prev_distance - current_distance
        print(prev_distance, current_distance)
        reward = delta_distance
        # set new observation
        self.set_target()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(self.current_yaw * 22.5 / 180 * np.pi),
                                            -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[2]])
        print("position after step", self.current_position)
        print("vec:", self.vector_observation)
        print("yaw:", self.current_yaw)
        self.observation[:, :, 0] = np.copy(self.range_image)
        self.observation[0, 0:3, 1] = np.copy(self.vector_observation)
        # check done
        #reward -= np.abs(self.current_yaw)*.5
        if self.current_position.x > 20:  # length of parkour
            done = True
        elif abs(self.current_position.y + 30) > 5:
            reward = -10
            done = True
        elif self.collision_flag:
            reward = -20
            done = True
        else:
            done = False
        info = {"current_position": self.current_position}
        #np.save("depth_data/im"+str(self.image_count), self.range_image)
        #self.image_count += 1
        print("reward:", reward)
        return self.observation, reward, done, info

    def reset(self):
        #if self.current_position.y < -30:
        self.go_position(self.current_position.x, self.current_position.y, 8)
        self.go_position(0, self.current_position.y, 8)
            #self.go_position(0, -29+np.random.randint(-2,3), 5)
        #else:
            #self.go_position(self.current_position.x, self.current_position.y, 5)
            #self.go_position(0, self.current_position.y, 5)
            #self.go_position(0, -29+np.random.randint(-2,3), 5)
        self.go_position(0, -30+np.random.randint(-2,3), 5)
        #self.go_position(0, -29, 5)
        self.collision_flag = False
        self.target_y = -29 # + np.random.randint(-1, 2)
        self.set_target()
        self.vector_observation = self.difference_between_points(self.target_position, self.current_position)
        self.observation[:, :, 0] = np.copy(self.range_image)
        self.observation[0, 0:3, 1] = np.copy(self.vector_observation)
        return self.observation

    def set_target(self):
        self.target_position = PoseStamped().pose.position
        self.target_position.x = self.current_position.x + 5
        self.target_position.y = self.target_y
        self.target_position.z = 5
        self.publish_target()

    def render(self, mode='human', close=False):
        pass

    def pose_callback(self, data):
        self.current_position = data.pose.position

    def range_image_callback(self, data):
        self.range_image = np.array(data.data).reshape((64, 64))/15.

    def model_name_callback(self, data):
        self.model_name = data.data

    def collision_callback(self, data):
        if data.data:
            self.collision_flag = True
            #print("colliiiddeeee")

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
        quat_z_yaw_dict = {-4:-0.7071068, -3:-0.5555702, -2:-0.3826834, -1: -0.1950903, 0: 0.0, 1: 0.1950903 , 2:0.3826834, 3:0.5555702, 4:0.7071068}
        quat_w_yaw_dict = {-4:0.7071068,  -3:0.8314696,  -2:0.9238795,  -1: 0.9807853,  0: 1.0, 1: 0.9807853 , 2:0.9238795, 3:0.8314696, 4:0.7071068}
        goal.pose.orientation.z = quat_z_yaw_dict[yaw]
        goal.pose.orientation.w = quat_w_yaw_dict[yaw]
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

    def distance_between_points(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return np.sqrt(x*x + y*y + z*z)

    def difference_between_points(self, p1, p2):
        return np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])


class AgiEnvVae_v2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AgiEnvVae_v2, self).__init__()

        # Gym elements
        self.action_space = gym.spaces.Discrete(7)
        self.latent_space_dimension = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_space_dimension + 3,), dtype=np.float64)
        self.action_dictionary = {0: (np.cos(22.5 / 180 * np.pi), np.sin(22.5 / 180 * np.pi), 1),
                                  1: (np.cos(22.5 / 180 * np.pi), np.sin(22.5 / 180 * np.pi), 0),
                                  2: (1, 0, 1),
                                  3: (1, 0, 0),
                                  4: (1, 0, -1),
                                  5: (np.cos(22.5 / 180 * np.pi), -np.sin(22.5 / 180 * np.pi), 0),
                                  6: (np.cos(22.5 / 180 * np.pi), -np.sin(22.5 / 180 * np.pi), -1),
                                  7: (0, 0, 2),
                                  8: (0, 0, -2), }
        self.current_position = PoseStamped().pose.position
        self.current_yaw = 0
        self.prev_yaw = 0

        self.range_vae = RangeVAE()
        self.range_image = np.ones((64, 64), dtype=np.float32)
        self.latent_range_image = np.ones((self.latent_space_dimension,), dtype=np.float32)
        self.collision_flag = False
        self.model_name = ""

        # ROS connection
        rospy.init_node('agi_gym_environment')
        self.r = rospy.Rate(10)
        self.ros_pub_pose = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.ros_pub_target = rospy.Publisher('target_position', PoseStamped, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/range_image_raw", Float32MultiArray, self.range_image_callback)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        self.r.sleep()
        print(self.model_name)
        rospy.Subscriber("/" + self.model_name + "/touch_sensor/value", BoolStamped, self.collision_callback)

        self.target_y = -30 + np.random.randint(-1, 2)
        self.set_target()
        self.r.sleep()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
            self.current_yaw * 22.5 / 180 * np.pi),
                                            -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                                                self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[2]])
        self.observation = np.zeros((self.latent_space_dimension + 3,))
        self.observation[3:] = np.copy(self.latent_range_image)
        self.observation[0:3] = np.copy(self.vector_observation)
        #print(self.vector_observation)
        self.r.sleep()
        self.reset()

        self.image_count = 0

    def step(self, discrete_action):
        # take action
        action = self.action_dictionary[discrete_action]
        print(action)
        self.go_position(
            self.current_position.x + action[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) - action[1] * np.sin(
                self.current_yaw * 22.5 / 180 * np.pi),
            self.current_position.y + action[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + action[1] * np.cos(
                self.current_yaw * 22.5 / 180 * np.pi), 5, yaw=self.current_yaw + action[2], check_collision=True)
        # calculate reward
        prev_distance = np.linalg.norm(self.vector_observation * np.array([1, 4, 1]))
        vo = self.difference_between_points(self.target_position, self.current_position)
        current_distance = np.linalg.norm(
            np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
                self.current_yaw * 22.5 / 180 * np.pi),
                      -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                          self.current_yaw * 22.5 / 180 * np.pi),
                      vo[2]]) * np.array([1, 4, 1]))
        delta_distance = prev_distance - current_distance
        reward = delta_distance
        reward -= np.abs(self.current_yaw) * .5
        # set new observation
        self.set_target()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(self.current_yaw * 22.5 / 180 * np.pi),
                                            -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[2]])
        self.latent_range_image = self.range_vae.range_vae(self.range_image)
        self.observation[3:] = np.copy(self.latent_range_image)
        self.observation[0:3] = np.copy(self.vector_observation)
        # check done
        if self.current_position.x > 15:
            done = True
        elif abs(self.current_position.y + 30) > 5:
            reward = -10
            done = True
        elif self.collision_flag:
            reward = -20
            done = True
        else:
            done = False
        info = {"current_position": self.current_position}
        #np.save("depth_data/im"+str(self.image_count), self.range_image)
        #self.image_count += 1
        print(self.observation)
        return self.observation, reward, done, info

    def reset(self):
        self.go_position(self.current_position.x, self.current_position.y, 8)
        self.go_position(0, self.current_position.y, 8)
        self.go_position(0, -30 + np.random.randint(-2, 3), 5)
        self.collision_flag = False
        self.target_y = -29 #+ np.random.randint(-1, 2)
        self.set_target()
        vo = self.difference_between_points(self.target_position, self.current_position)
        self.vector_observation = np.array([vo[0] * np.cos(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.sin(
            self.current_yaw * 22.5 / 180 * np.pi),
                                            -vo[0] * np.sin(self.current_yaw * 22.5 / 180 * np.pi) + vo[1] * np.cos(
                                                self.current_yaw * 22.5 / 180 * np.pi),
                                            vo[2]])
        self.observation[3:] = np.copy(self.latent_range_image)
        self.observation[0:3] = np.copy(self.vector_observation)
        return self.observation

    def set_target(self):
        self.target_position = PoseStamped().pose.position
        self.target_position.x = self.current_position.x + 5
        self.target_position.y = self.target_y
        self.target_position.z = 5
        self.publish_target()

    def render(self, mode='human', close=False):
        pass

    def pose_callback(self, data):
        self.current_position = data.pose.position

    def range_image_callback(self, data):
        self.range_image = np.array(data.data).reshape((64*64, ))

    def model_name_callback(self, data):
        self.model_name = data.data

    def collision_callback(self, data):
        if data.data:
            self.collision_flag = True
            #print("colliiiddeeee")

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
        quat_z_yaw_dict = {-4:-0.7071068, -3:-0.5555702, -2:-0.3826834, -1: -0.1950903, 0: 0.0, 1: 0.1950903 , 2:0.3826834, 3:0.5555702, 4:0.7071068}
        quat_w_yaw_dict = {-4:0.7071068,  -3:0.8314696,  -2:0.9238795,  -1: 0.9807853,  0: 1.0, 1: 0.9807853 , 2:0.9238795, 3:0.8314696, 4:0.7071068}
        goal.pose.orientation.z = quat_z_yaw_dict[yaw]
        goal.pose.orientation.w = quat_w_yaw_dict[yaw]
        self.prev_yaw = self.current_yaw
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

    def distance_between_points(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return np.sqrt(x*x + y*y + z*z)

    def difference_between_points(self, p1, p2):
        return np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])


class RangeVAE():

    def __init__(self):
        def xavier(in_shape):
            return tf.random_normal(shape=in_shape, stddev=1. / tf.sqrt(in_shape[0] / 2.))

        image_dimension = 4096
        neural_network_dimension = 512
        latent_var_dimension = 5

        self.Weight = {"weight_matrix_encoder_hidden": tf.Variable(xavier([image_dimension, neural_network_dimension])),
                  "weight_mean_hidden": tf.Variable(xavier([neural_network_dimension, latent_var_dimension])),
                  "weight_std_hidden": tf.Variable(xavier([neural_network_dimension, latent_var_dimension])),
                  "weight_matrix_decoder_hidden": tf.Variable(xavier([latent_var_dimension, neural_network_dimension])),
                  "weight_decoder": tf.Variable(xavier([neural_network_dimension, image_dimension]))
                  }

        self.Bias = {"bias_matrix_encoder_hidden": tf.Variable(xavier([neural_network_dimension])),
                "bias_mean_hidden": tf.Variable(xavier([latent_var_dimension])),
                "bias_std_hidden": tf.Variable(xavier([latent_var_dimension])),
                "bias_matrix_decoder_hidden": tf.Variable(xavier([neural_network_dimension])),
                "bias_decoder": tf.Variable(xavier([image_dimension]))
                }

        self.image_X = tf.placeholder(tf.float32, shape=[None, image_dimension])

        self.encoder_layer = tf.add(tf.matmul(self.image_X, self.Weight["weight_matrix_encoder_hidden"]),
                               self.Bias["bias_matrix_encoder_hidden"])
        self.encoder_layer = tf.nn.tanh(self.encoder_layer)

        self.mean_layer = tf.add(tf.matmul(self.encoder_layer, self.Weight["weight_mean_hidden"]), self.Bias["bias_mean_hidden"])
        self.std_layer = tf.add(tf.matmul(self.encoder_layer, self.Weight["weight_std_hidden"]), self.Bias["bias_std_hidden"])

        self.epsilon = tf.random_normal(tf.shape(self.std_layer), dtype=tf.float32, mean=0.0, stddev=1.0)

        self.latent_layer = self.mean_layer + tf.exp(0.5 * self.std_layer) * self.epsilon

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver.restore(self.sess, "./vae_model/model5.ckpt")

    def range_vae(self, cv_image):
        x_sample = cv_image.reshape((1,4096))/15.
        latent_repr = self.sess.run(self.latent_layer, feed_dict={self.image_X: x_sample})
        return latent_repr

