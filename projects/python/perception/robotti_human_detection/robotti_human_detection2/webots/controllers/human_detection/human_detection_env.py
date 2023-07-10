from controller import Supervisor, Robot, Motor, Camera, Lidar, LidarPoint, GPS, CameraRecognitionObject

#import sys
import base64
from typing import List

from abc import ABC
import numpy as np
import gym

from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.target import BoundingBoxList
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray

MAX_SPEED = 6.28

class Env(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        super(Env, self).__init__()
        
        if '--cuda' in args:
          self.cudaEnabled = True

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.camera = self.robot.getDevice('front_bottom_camera')
        self.camera.enable(self.timestep)
        self.display = self.robot.getDevice('recognition display')
        self.display.attachCamera(self.camera)
        self.displaySize = [self.display.getWidth(), self.display.getHeight()]
        self.recognizer, self.detector = None, None
        self.gps = self.robot.getDevice("Hemisphere_v500")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        
        self.left_motor_front = self.robot.getDevice("left_front_wheel_joint_motor")
        self.left_motor_rear = self.robot.getDevice("left_rear_wheel_joint_motor")
        self.right_motor_front = self.robot.getDevice("right_front_wheel_joint_motor")
        self.right_motor_rear = self.robot.getDevice("right_rear_wheel_joint_motor")
        self.left_motor_front.setPosition(float('inf'))
        self.left_motor_rear.setPosition(float('inf'))
        self.right_motor_front.setPosition(float('inf'))
        self.right_motor_rear.setPosition(float('inf'))
        
        self.left_motors = [self.left_motor_front, self.left_motor_rear]
        self.right_motors = [self.right_motor_front, self.right_motor_rear]
        self.motors = self.left_motors + self.right_motors
        self.move_action = 'forwards'
        self.move_last_action = self.move_action
        self.move_last_value = 0
        self.load_models()
        self.forwards(0.2)
        self.isStopped = False
        self.steps_from_detection = 5

    def load_models(self):
        if self.cudaEnabled:
            device = 'cuda'
        else:
            device = 'cpu'
        self.detector = YOLOv5DetectorLearner(model_name='yolov5x', device=device)

    def reset(self):
        self.robot.step(self.timestep)
        obs = self.get_obs()
        self.robot.step(self.timestep)
        return obs

    def step(self, action):
        done = False
        reward = 0
        self.confidence = 0.0
        if action == 0:
            # Stay
            self.robot.step(self.timestep)
            obs = self.get_obs()
            reward = self.get_reward(action)

        obs = self.detect(obs)
        
        if not self.isStopped:
            if self.move_action == 'forwards':
                if self.gps.getValues()[0] > 2.5:
                    self.move_last_action = self.move_action
                    self.move_action = 'turnLeft'
                    self.turn('left')
                elif self.gps.getValues()[0] < -44:
                    self.move_last_action = self.move_action
                    self.move_action = 'turnRight'
                    self.turn('right')
            elif self.move_action == 'forwardsLeft' or self.move_action == "forwardsRight":
                if abs(self.gps.getValues()[1] - self.move_last_value) > 3.2:
                    self.turn(self.move_action.lstrip('forwards').lower())
                    self.move_last_action = self.move_action
                    self.move_action = "turn" + self.move_action.lstrip('forwards')
            else:
                # check that rotation is completed
                if self.move_action == 'turnLeft':
                    if self.move_last_action != 'forwardsLeft' and self.compass.getValues()[0] > 0.999:
                        self.move_last_action = self.move_action
                        self.move_action = "forwards" + self.move_action.lstrip('turn')
                        self.forwards(0.1)
                        self.move_last_value = self.gps.getValues()[1]
                    elif self.compass.getValues()[1] < -0.999:
                        self.move_last_action = self.move_action
                        self.move_action = "forwards"
                        self.forwards(0.2) 
                elif self.move_action == 'turnRight':
                    if self.move_last_action != 'forwardsRight' and self.compass.getValues()[0] > 0.999:
                        self.move_last_action = self.move_action
                        self.move_action = "forwards" + self.move_action.lstrip('turn')
                        self.forwards(0.1)
                        self.move_last_value = self.gps.getValues()[1]
                    elif self.compass.getValues()[1] > 0.999:
                        self.move_last_action = self.move_action
                        self.move_action = "forwards"
                        self.forwards(0.2)

        return obs, reward, done, {}

    def get_obs(self):
        cameraData = self.camera.getImage()
        if cameraData:
            frame = np.frombuffer(cameraData, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]
            return frame

    def get_reward(self, action):
        reward = 0
        return reward

    def detect(self, img):
        if self.steps_from_detection < 10:
            self.steps_from_detection += 1
            
        # clear previous annotations on display
        self.display.setAlpha(0)
        self.display.fillRectangle(0, 0, self.displaySize[0], self.displaySize[1])

        class_names = self.detector.classes
        boxes = self.detector.infer(img)
        bounding_boxes = BoundingBoxListToNumpyArray()(boxes)
        if len(bounding_boxes) == 0:
            self.sendImage()
            return
        boxes = bounding_boxes[:, :4]
        classes = bounding_boxes[:, 5].astype(np.int)
        for idx, pred_box in enumerate(boxes):
            pred_box_w, pred_box_h = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
            if pred_box_w > 200 or pred_box_h > 200:
                continue
            valid = pred_box_w > 15 or pred_box_h > 15
            if valid:
                self.steps_from_detection = 0

            c1 = (max(0, int(pred_box[0] * 0.5)), max(0, int(pred_box[1] * 0.5)))
            c2 = (min(img.shape[1] * 0.5, int(pred_box[2] * 0.5)), min(img.shape[0] * 0.5, int(pred_box[3] * 0.5)))

            self.display.setAlpha(1)
            self.display.setColor(0x0000FF)
            self.display.drawRectangle(c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1])

            if class_names is not None:
                label = "{}".format(class_names[classes[idx]])
                self.display.fillRectangle(c1[0], c2[1] - 20, 60, 20)
                self.display.setColor(0xFFFFFF)
                self.display.setFont('Arial', 10, True)
                self.display.drawText(label, c1[0] + 2, c2[1] - 15)
        self.sendImage()
        if self.isStopped and self.steps_from_detection >= 5:
            self.go()
        elif not self.isStopped and self.steps_from_detection < 5:
            self.stop()


    def turn(self, direction):
        self.robot.wwiSendText('log:Turn ' + direction + '.')
        left_speed = -0.1 if direction == 'left' else 0.1
        right_speed = 0.1 if direction == 'left' else -0.1
        for motor in self.left_motors:
            motor.setVelocity(left_speed * MAX_SPEED)
        for motor in self.right_motors:
            motor.setVelocity(right_speed * MAX_SPEED)

    def forwards(self, speed):
        self.robot.wwiSendText('log:Move forwards.')
        for motor in self.motors:
            motor.setVelocity(speed * MAX_SPEED)

    def stop(self):
        self.robot.wwiSendText('log:Stop.')
        self.previousVelocities = []
        self.isStopped = True
        for motor in self.motors:
            self.previousVelocities.append(motor.getVelocity())
            motor.setVelocity(0)
    
    def go(self):
        self.robot.wwiSendText('log:Go.')
        self.isStopped = False
        i = 0
        for motor in self.motors:
            motor.setVelocity(self.previousVelocities[i])
            i += 1
            
    def sendImage(self):
        self.display.imageSave(None, 'display.jpg')
        with open('display.jpg', 'rb') as f:
            fileString = f.read()
            fileString64 = base64.b64encode(fileString).decode()
            self.robot.wwiSendText('display-image:data:image/jpeg;base64,' + fileString64)
