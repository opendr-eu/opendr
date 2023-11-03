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
import math
import random

import gym
import numpy as np
import torch

from abc import ABC
from transforms3d import quaternions
from gym import spaces

from controller import Supervisor

from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.transforms import \
    BoundingBoxListToNumpyArray
from opendr.perception.face_recognition import FaceRecognitionLearner


class Env(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space=10, seed=0):
        super(Env, self).__init__()

        # Webots Environment
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.actions = action_space
        self.robot = Supervisor()
        self.mavic = self.robot.getSelf()
        self.rotation = self.mavic.getField('rotation')
        self.position = self.mavic.getField('translation')
        self.timestep = int(self.robot.getBasicTimeStep())
        self.root = self.robot.getRoot()
        self.children = self.root.getField('children')
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.reco_camera = self.robot.getDevice('reco_camera')
        self.reco_camera.disable()
        self.robot.keyboard.enable(self.timestep)
        self.front_left_led = self.robot.getDevice('front left led')
        self.front_right_led = self.robot.getDevice('front right led')
        self.imu = self.robot.getDevice('inertial unit')
        self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.robot.keyboard.enable(self.timestep)
        self.sensor = self.robot.getDevice('touch sensor')
        self.sensor.enable(self.timestep)
        # ----------------------------- GYM STUFF ----------------------------- #
        self.action_space = spaces.Discrete(self.actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 400, 3), dtype=np.uint8)
        self.reward_range = [-100, 100]
        self.metadata = None
        self.step_counter = 0
        self.num_envs = 1
        self.distance, self.angle, self.angle_frontal, self.height = None, None, None, None
        self.prev_distance, self.prev_angle, self.prev_angle_frontal, self.prev_height = 12, 1, 1, 2
        self.confidence = 0.0
        self.human_position = None
        self.recognizer, self.detector = None, None
        self.load_models()
        self.target_distance = 1.0
        self.target_height = 1.7
        self.distance_tolerance = 0.3
        self.height_tolerance = 0.3
        self.eval = False
        self.info = {}
        self.fr_eval = False
        self.located = False

    def load_models(self):
        self.recognizer = FaceRecognitionLearner(device='cuda', backbone='mobilefacenet', mode='backbone_only')
        self.recognizer.download('./fr_model')
        self.recognizer.load('./fr_model')
        self.recognizer.fit_reference('./data/images', './reference_lab', create_new=True)
        self.detector = RetinaFaceLearner(backbone='mnet', device='cuda')
        self.detector.download(".", mode="pretrained")
        self.detector.load("./retinaface_mnet")

    def reset(self, seed=None):
        self.step_counter = 0
        self.located = False
        self.robot.simulationReset()
        self.robot.simulationResetPhysics()
        self.initialize_world()
        self.robot.step(self.timestep)
        self.replace_humans()
        self.robot.step(self.timestep)
        self.rotation.setSFRotation([0, 0, 1, random.randint(0, 50) * 0.1309])
        position = [random.randint(-8, 8), random.randint(-8, 8), random.uniform(0.5, 2.5)]
        self.position.setSFVec3f(position)
        self.robot.step(self.timestep)
        for i in range(random.randint(1, 5)):
            self.robot.step(self.timestep)
            self.get_reward(0)
            if self.sensor.getValue() != 0.0 or self.distance < 0.2 or self.angle < 0.1:
                self.reset()
        obs = self.get_obs()
        return obs

    def step(self, action):
        if self.angle < 0.5:
            self.located = True
        info = {"TimeLimit.truncated": False}
        self.step_counter += 1
        done = False
        reward = 0
        self.confidence = 0.0
        if action == 0:
            # Stay
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 1:
            # Move forwards 10cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([0.1, 0, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 2:
            # Move backwards 10cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([-0.1, 0, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 3:
            # Strafe left 10cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([0, 0.1, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 4:
            # Strafe right 10cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([0, -0.1, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 5:
            # Rotate clockwise 3°
            rotation = self.rotation.getSFRotation()
            q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
            q2 = quaternions.axangle2quat([0, 0, 1], 3 * 3.14 / 180)
            q = quaternions.qmult(q1, q2)
            vec, angle = quaternions.quat2axangle(q)
            new_rotation = [vec[0], vec[1], vec[2], angle]
            self.rotation.setSFRotation(new_rotation)
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)
            if not self.located:
                reward = abs(reward)

        if action == 6:
            # Rotate counterclockwise 3°
            rotation = self.rotation.getSFRotation()
            q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
            q2 = quaternions.axangle2quat([0, 0, 1], -3 * 3.14 / 180)
            q = quaternions.qmult(q1, q2)
            vec, angle = quaternions.quat2axangle(q)
            new_rotation = [vec[0], vec[1], vec[2], angle]
            self.rotation.setSFRotation(new_rotation)
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)
            if not self.located:
                reward = -abs(reward)

        if action == 7:
            # Deploy face detection and recognition module
            self.reco_camera.enable(self.timestep)
            self.robot.step(self.timestep)
            obs = self.get_obs()
            frame = self.get_reco_obs()
            self.confidence = self.detect_and_recognize(frame)
            if (math.isclose(self.distance, self.target_distance, abs_tol=self.distance_tolerance) and
                    self.angle_frontal < 0.1 and self.angle < 0.1 and
                    math.isclose(self.height, self.target_height, abs_tol=self.height_tolerance)):
                reward = self.confidence * 20
                done = True
            else:
                reward = -0.05
            self.reco_camera.disable()

        if action == 8:
            # Move upwards 5cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([0.0, 0, 0.05])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if action == 9:
            # Move downwards 5cm
            ori = np.array(self.mavic.getOrientation()).reshape([3, 3])
            b = np.array([0, 0, -0.05])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        if self.step_counter == 2000:
            reward = 0
            done = True
            info = {"TimeLimit.truncated": True}

        if self.sensor.getValue() != 0.0 or self.distance < 0.2:
            done = True
            reward = -1

        return obs, reward, done, info

    def get_obs(self):
        cameraData = self.camera.getImage()
        if cameraData:
            frame = np.frombuffer(cameraData, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]
            return frame

    def get_reco_obs(self):
        cameraData = self.reco_camera.getImage()
        if cameraData:
            frame = np.frombuffer(cameraData, np.uint8).reshape(
                (self.reco_camera.getHeight(), self.reco_camera.getWidth(), 4))
            frame = frame[:, :, :3]
            return frame

    def get_reward(self, action):
        reward = 0
        human_pos = np.array(self.human_position.getSFVec3f())
        drone_pos = np.array(self.position.getSFVec3f())
        human_pos[2] = drone_pos[2]
        drone_person = np.array(drone_pos - human_pos)
        distance = np.linalg.norm(drone_person)
        drone_person_vector = drone_person / np.linalg.norm(drone_person)
        ori_drone = np.array(self.mavic.getOrientation()).reshape([3, 3])
        b = np.array([1, 0, 0])
        new_p = ori_drone.dot(b) + drone_pos
        drone_forward = drone_pos - new_p
        drone_forward = drone_forward / np.linalg.norm(drone_forward)
        unit_vector = drone_forward.dot(drone_person_vector)
        angle = np.arccos(unit_vector)
        height = drone_pos[2]
        human_pos[2] = drone_pos[2]
        ori_person = np.array(self.human.getOrientation()).reshape([3, 3])
        c = np.array([1, 0, 0])
        new_p = ori_person.dot(c) + human_pos
        human_forward = human_pos - new_p
        human_forward = human_forward / np.linalg.norm(human_forward)
        unit_vector_2 = drone_forward.dot(human_forward)
        angle_frontal = np.arccos(-unit_vector_2)
        if math.isnan(angle):
            angle = np.arccos(-1)
        if math.isnan(angle_frontal):
            angle_frontal = np.arccos(-1)
        if not self.located:
            reward += self.get_a_reward(angle) - abs(self.get_d_reward(distance))
        else:
            reward += self.get_a_reward(angle) + self.get_d_reward(distance)
        reward += self.get_h_reward(height) + self.get_b_reward(angle_frontal)
        self.distance = distance
        self.angle = angle
        self.angle_frontal = angle_frontal
        self.height = height
        if self.eval:
            reward = 0
        return reward

    def get_a_reward(self, angle):
        a_reward = 2 * (self.prev_angle - angle)
        self.prev_angle = angle
        return a_reward

    def get_b_reward(self, angle_frontal):
        b_reward = 2 * (self.prev_angle_frontal - angle_frontal)
        self.prev_angle_frontal = angle_frontal
        return b_reward

    def get_d_reward(self, distance):
        if distance > self.target_distance:
            d_reward = self.prev_distance - distance
        else:
            d_reward = distance - self.prev_distance
        self.prev_distance = distance
        return d_reward

    def get_h_reward(self, height):
        if height > self.target_height:
            h_reward = self.prev_height - height
        else:
            h_reward = height - self.prev_height
        self.prev_height = height
        return h_reward

    def set_eval(self):
        self.eval = True

    def unset_eval(self):
        self.eval = False

    def detect_and_recognize(self, frame):
        result = None
        confidence = 0.0
        bounding_boxes = self.detector.infer(frame)
        if bounding_boxes:
            bounding_boxes_ = BoundingBoxListToNumpyArray()(bounding_boxes)
            boxes = bounding_boxes_[:, :4]
            for idx, box in enumerate(boxes):
                (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face_img = frame[startY:endY, startX:endX]
                result = self.recognizer.infer(face_img)
                if result.description != 'Not found':
                    break
            if not result:
                confidence = 0.0
            else:
                confidence = result.confidence
        return confidence

    def initialize_world(self):
        self.robot.step(self.timestep)
        sides = ['bed', 'radiator_1', 'fridge', 'oven',
                 'round_table']
        faces = ['clock', 'radiator_2', 'door', 'cabinet', 'board', 'table', 'tree_1']
        living_room = ['armchair_1', 'armchair_2', 'sofa', 'carpet']
        desk = ['chair_1', 'chair_2', 'chair_3', 'chair_4']
        trees = ['tree_2', 'tree_3', 'tree_4', 'tree_5']

        ########################################
        # Change side objects position #########
        ########################################
        seeds = []
        for item in sides:
            seed = random.randint(-9, 9)
            seeds.append(seed)
            while seed in seeds:
                seed = random.randint(-9, 9)
            item = self.robot.getFromDef(item)
            position = item.getField('translation')
            vec = position.getSFVec3f()
            position.setSFVec3f([seed, vec[1], vec[2]])
            vec = position.getSFVec3f()
        ########################################
        # Change face objects position #########
        ########################################
        seeds = []
        for item in faces:
            seed = random.randint(-8, 8)
            seeds.append(seed)
            while seed in seeds:
                seed = random.randint(-8, 8)
            item = self.robot.getFromDef(item)
            position = item.getField('translation')
            vec = position.getSFVec3f()
            position.setSFVec3f([vec[0], seed, vec[2]])
        ########################################
        # Change living room position ##########
        ########################################
        round_table = self.robot.getFromDef('round_table')
        position = round_table.getField('translation')
        vec = position.getSFVec3f()
        translation = [[2, +2, 0], [-2, 2, 0], [0, 3, 0], [0, 1.5, 0]]
        for cnt, item in enumerate(living_room):
            item = self.robot.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([vec[0] + translation[cnt][0], vec[1] + translation[cnt][1], vec[2]])
        #################################
        # Change Desk Position ##########
        #################################
        table = self.robot.getFromDef('table')
        position = table.getField('translation')
        vec = position.getSFVec3f()
        translation = [[-1, 0.5, 0], [-1, -0.5, 0], [1, 0.5, 0], [1, -0.5, 0]]
        for cnt, item in enumerate(desk):
            item = self.robot.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([vec[0] + translation[cnt][0], vec[1] + translation[cnt][1], vec[2]])
        #################################
        # Change Tree Position ##########
        #################################
        tree = self.robot.getFromDef('tree_1')
        position = tree.getField('translation')
        vec = position.getSFVec3f()
        translation = [[0, 1, 0], [0, -1, 0], [1, 0.5, 0], [1, -0.5, 0]]
        for cnt, item in enumerate(trees):
            item = self.robot.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([vec[0] + translation[cnt][0], vec[1] + translation[cnt][1], vec[2]])
        self.robot.step(self.timestep)

    def replace_humans(self):
        rnd = random.randint(1, 10)
        self.robot.step(self.timestep)
        if rnd < 10:
            human_model = 'human_0' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.robot.getFromDef(human_model)
        else:
            human_model = 'human_' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.robot.getFromDef(human_model)
        self.robot.step(self.timestep)
        self.human_position = self.human.getField('translation')
        self.human_position.setSFVec3f([random.randrange(-7, 7), random.randrange(-7, 7), 0])
        rotation = self.human.getField('rotation')
        rotation.setSFRotation([0, 0, 1, random.randint(0, 50) * 0.1309])
