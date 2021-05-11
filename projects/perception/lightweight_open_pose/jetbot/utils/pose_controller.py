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

import time
from utils.pose_utils import calculate_horizontal_offset, calculate_upper_body_height, calculate_body_area
from utils.pid import PID
import numpy as np
import cv2


class PoseController:
    def __init__(self, robot, pose_estimator, visualization_handler_fn=None, fall_handler_fn=None, active=True,
                 infer_delay=0, disable_collision=False):
        """
        Initializes the Executer class responsible for detecting humans and issuing the appropriate control commands.
        Note that the current implementation assumes that only one human appears in the scene. If more than one appears,
        then the robot will only process information regarding the first one detected.
        @param robot: The robot controller to be used for issuing the control commands
        @param pose_estimator: The pose estimator to be used for fall detection
        @param visualization_handler_fn: A function with signature visualization_handler_fn(img, pose, statistics) that
        can be used for visualization the detections and other information regarding the robot/target. This is an optional
        function. Please note that the call to this function is blocking. Non-blocking functionality should be implemented
        by the visualization_handler_fn() function.
        @param fall_handler_fn: function to call when a fall is detected
        @param active: enables active perception
        @param infer_delay: delay after each inference operation (used to fully simulate the real hardware)
        @param disable_collision: disables collision avoidance (to enable faster execution)
        """

        self.active = active
        self.robot = robot
        self.pose_estimator = pose_estimator
        self.visualization_handler = visualization_handler_fn
        self.fall_handler_fn = fall_handler_fn
        self.infer_delay = infer_delay

        if disable_collision:
            self.enable_depth_perception = False
        else:
            # Enables some very basic collision detection
            self.enable_depth_perception = True
            self.collision_depth_threshold = 0.06
            import gluoncv
            import mxnet as mx
            self.ctx = mx.gpu(0)
            self.collision_model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_stereo_640x192',
                                                               pretrained_base=False, ctx=self.ctx, pretrained=True)

        # Webots constants
        if self.robot.robot_interface == 'webots':

            # Amount of movement for one wheel when performing a rotation
            self.rotation_interval = 0.5

            # Used to control the limits within the human should be in order to considered centered
            self.translation_offset_limit = 0.04

            # PID controller to center the subject
            self.rotation_pid = PID(1, 0, 0.1, output_limits=(-3, 3), setpoint=0, cutoff=0.01)

            # Distances (calculated as the areas covered by the subject in the current frame) to keep from a subject
            self.max_distance_size = 0.07
            self.min_distance_size = 0.12
            self.distance_pid = PID(100, 0, 0, output_limits=(-2, 2), setpoint=self.max_distance_size)

            # Number of maximum missed joints allowed for assuming this is a clean pose
            self.min_joints_threshold = 8

            # Number of frames that can be passed with no detection, before initiating rotate_to_detect()
            self.patience = 5

            # Camera threshold for assessing a fall
            self.fall_threshold = 0.4

            # (Max) distance to be covered when a fall is detected
            self.distance_fall = 10

            #  Distance to move forward for active perception
            self.active_step_forward = 2

            # Number of successive detection before considering a detection stable
            self.active_successive_limit = 5
            # Active control PIDs
            self.rotation_pid_active = PID(2, 0, 0.1, output_limits=(-3, 3), setpoint=0, cutoff=0.01)
            self.distance_pid_active = PID(30, 0, 0, output_limits=(-5, 5), setpoint=0.1)
        elif self.robot.robot_interface == 'jetbot':
            # Amount of movement for one wheel when performing a rotation
            self.rotation_interval = 0.5

            # Used to control the limits within the human should be in order to considered centered
            self.translation_offset_limit = 0.04

            # PID controller to center the subject
            self.rotation_pid = PID(1, 0, 0.1, output_limits=(-3, 3), setpoint=0, cutoff=0.01)

            # Distances (calculated as the areas covered by the subject in the current frame) to keep from a subject
            self.max_distance_size = 0.07
            self.min_distance_size = 0.12
            self.distance_pid = PID(100, 0, 0, output_limits=(-2, 2), setpoint=self.max_distance_size)

            # Number of maximum missed joints allowed for assuming this is a clean pose
            self.min_joints_threshold = 8

            # Number of frames that can be passed with no detection, before initiating rotate_to_detect()
            self.patience = 5

            # Camera threshold for assessing a fall
            self.fall_threshold = 0.4

            # (Max) distance to be covered when a fall is detected
            self.distance_fall = 10

            #  Distance to move forward for active perception
            self.active_step_forward = 2

            # Number of successive detection before considering a detection stable
            self.active_successive_limit = 5
            # Active control PIDs
            self.rotation_pid_active = PID(2, 0, 0.1, output_limits=(-3, 3), setpoint=0, cutoff=0.01)
            self.distance_pid_active = PID(30, 0, 0, output_limits=(-5, 5), setpoint=0.1)
        else:
            assert False

        # Cache
        self.last_img = None
        self.last_pose = None

        # Using a running average to smooth the size
        self.running_average_size = 0
        self.size_smoothing = 0.3

        # Use a running average to check if fall has been detected
        self.running_average_fall = 0
        self.fall_smoothing = 0.9
        self.confident_fall_threshold = 0.9

        # Number of finetuning movements to perform during rotate and detect phase
        self.max_rotates_finetuning = 3

        # Limit for the determining whether there is something interesting in the heat map, as well as its relative size
        self.active_detection_limit = 0.25
        self.active_size_limit = 0.4

        self.image_width = 800
        self.image_height = 600

    def rotate_to_detect(self):
        """
        Rotates the robot until finding a human target and then returns
        """

        # Get a frame and check for detections
        self.last_img = self.robot.get_camera_data()
        self.last_pose = None
        poses = self.pose_estimator.infer(self._get_infer_image())
        self.wait()

        if self.active:

            # offset_x is used to determine the direction of movements
            offset_x = None
            heatmap = None

            # Counter of stable detections (used to stop the active perception)
            counter = 0
            while True:
                control_left, control_right = 0, 0

                # if offset has been detected, move to the direction that will bring us closer to the subject
                if offset_x is not None:
                    # Calculate the rotation of the robot to center the target
                    offset_command = self.rotation_pid_active(offset_x)
                    if offset_command > 0:
                        control_left += np.abs(offset_command)
                    else:
                        control_right += np.abs(offset_command)
                else:
                    offset_command = None

                self.last_img = self.robot.get_camera_data()

                heatmap, poses = self.pose_estimator.infer_active(self.last_img)
                self.wait()

                # Get the probability that a pixel is not a joint
                heatmap = (1 - heatmap[:, :, -1])
                max_conf = np.max(heatmap)
                heatmap = cv2.resize(heatmap, (200, 160))
                person_area = np.sum([heatmap > self.active_size_limit]) / (200 * 160)
                # Convert heatmap to image
                heatmap = np.uint8((heatmap) * 255)

                # If something interesting has been detected
                if max_conf > self.active_detection_limit:
                    # Locate the maximum in order to decide where to move in the next step
                    i, j = np.unravel_index(np.argmax(heatmap), np.array(heatmap).shape)
                    offset_x = (float(j) / heatmap.shape[1]) - 0.5
                    # Also, we can now move closer to the point of interest

                    distance_command = self.distance_pid_active(person_area)
                    control_left += distance_command
                    control_right += distance_command

                else:
                    offset_x = None

                def visualization_fn(img):
                    return self.visualization_handler(img, self.last_pose,
                                                      {'state': 'rotate_to_detect_active',
                                                       'heatmap': heatmap,
                                                       'control_left': control_left,
                                                       'control_right': control_right,
                                                       'size': person_area, 'offset': offset_x,
                                                       })

                # Perform the actions
                if offset_command is None:
                    self.robot.rotate(self.rotation_interval, visualization_fn)
                else:
                    self.robot.rotate(offset_command, visualization_fn)

                if max_conf > self.active_detection_limit:
                    self.safe_translate(distance_command, visualization_fn)

                # If we have successfully detected a pose and we have a stable detection we can end this process
                if len(poses) > 0 and np.sum(poses[0].data == -1) < self.min_joints_threshold:
                    counter += 1
                else:
                    counter = 0

                if counter > self.active_successive_limit:
                    break
        else:
            # Slowly rotate until detecting a human
            while len(poses) == 0:
                def visualization_fn(img):
                    return self.visualization_handler(img, self.last_pose,
                                                      {'state': 'rotate_to_detect_active',
                                                       })

                self.robot.rotate(self.rotation_interval, visualization_fn)
                self.last_img = self.robot.get_camera_data()
                poses = self.pose_estimator.infer(cv2.resize(self.last_img, (600, 800)))
                self.wait()
                if len(poses) > 0:
                    self.last_pose = poses[0]
                    offset_center = calculate_horizontal_offset(poses[0], self.image_width)

                    def visualization_fn(img):
                        return self.visualization_handler(img, self.last_pose,
                                                          {'state': 'rotate_to_detect_active',
                                                           'offset': offset_center,
                                                           })

                    if offset_center > 0:
                        self.robot.rotate(-self.rotation_interval, visualization_fn)
                    else:
                        self.robot.rotate(self.rotation_interval, visualization_fn)
                    break
                else:
                    self.last_pose = None

    def _get_infer_image(self):
        return cv2.resize(self.last_img, (self.image_width, self.image_height))

    def monitor_target(self):
        """
        Centers a target and tries to keep an appropriate distance.
        Periodically checks for falls and enables the fall mitigation routine.
        @return:
        @rtype:
        """

        # Counter for the number of frames with no detection
        no_detection_frames = 0
        control_left, control_right = 0, 0
        fall = False
        offset_center, size = 0, 0

        while no_detection_frames < self.patience:

            self.last_img = self.robot.get_camera_data()
            poses = self.pose_estimator.infer(self._get_infer_image())
            self.wait()

            if len(poses) > 0:
                # Reset counter and keep the last pose

                self.last_pose = poses[0]

                # Check the quality of the pose
                if np.sum(self.last_pose.data == -1) >= self.min_joints_threshold:
                    no_detection_frames += 1
                    self.visualization_handler(self.last_img, None, {'state': 'monitor_target',
                                                                     'control_left': control_left,
                                                                     'control_right': control_right, 'fall': fall,
                                                                     'size': size, 'offset': offset_center,
                                                                     'fall_confidence': self.running_average_fall,
                                                                     'skipped': True})
                    self.robot.step()
                else:
                    no_detection_frames = 0
                    # Appropriate control the robot to keep the target centered and within appropriate distance
                    # Keep some statistics for visualization
                    control_left, control_right = 0, 0

                    height = calculate_upper_body_height(self.last_pose, self.image_height)
                    size_scaler = 1

                    # If human is on ground, initially account for the distance discrepancy
                    if height > self.fall_threshold:
                        self.running_average_fall = self.running_average_fall * self.fall_smoothing + (
                                1 - self.fall_smoothing)
                        size_scaler = 2
                    else:
                        self.running_average_fall = self.running_average_fall * self.fall_smoothing

                    # Calculate the rotation of the robot to center the target
                    offset_center = calculate_horizontal_offset(self.last_pose, self.image_width)
                    offset_command = self.rotation_pid(offset_center)
                    if offset_command > 0:
                        control_left += np.abs(offset_command)
                    else:
                        control_right += np.abs(offset_command)

                    # Calculate the distance in order to have the robot in a comfortable distance
                    size = calculate_body_area(self.last_pose, self.image_width, self.image_height) * size_scaler
                    if self.running_average_size == 0:
                        self.running_average_size = size
                    self.running_average_size = self.size_smoothing * self.running_average_size + (
                            1 - self.size_smoothing) * size

                    if self.running_average_size < self.max_distance_size or self.running_average_size > self.min_distance_size:
                        distance_command = self.distance_pid(self.running_average_size)
                        control_left += distance_command
                        control_right += distance_command
                    else:
                        distance_command = 0

                    # Check if we had detected a fall with enough confidence
                    if self.running_average_fall > self.confident_fall_threshold:
                        fall = True
                    else:
                        fall = False

                    def visualization_handler(img):
                        return self.visualization_handler(img, self.last_pose,
                                                          {'state': 'monitor_target',
                                                           'control_left': control_left,
                                                           'control_right': control_right,
                                                           'fall': fall,
                                                           'size': size,
                                                           'offset': offset_center,
                                                           'fall_confidence': self.running_average_fall,
                                                           'skipped': False, 'control': True})

                    self.last_img = self.robot.get_camera_data()
                    self.visualization_handler(self.last_img, self.last_pose, {'state': 'monitor_target',
                                                                               'control_left': control_left,
                                                                               'control_right': control_right,
                                                                               'fall': fall,
                                                                               'size': size, 'offset': offset_center,
                                                                               'fall_confidence': self.running_average_fall,
                                                                               'skipped': False,
                                                                               'fall': fall})

                    # Center the subject
                    if np.abs(offset_command) > 0.01:
                        self.robot.rotate(offset_command, visualization_handler)

                    # Control the distance
                    if distance_command != 0:
                        self.safe_translate(distance_command, visualization_handler)
                    self.last_img = self.robot.get_camera_data()
                    self.robot.step()

                    # Handle a fall
                    if fall:
                        self.handle_fall()
                        # Reset threshold to allow fast recovery
                        self.running_average_fall = 0

            else:
                self.last_pose = None
                no_detection_frames += 1
                self.robot.step()
            self.visualization_handler(self.last_img, self.last_pose, {'state': 'monitor_target',
                                                                       'control_left': control_left,
                                                                       'control_right': control_right, 'fall': fall,
                                                                       'size': size, 'offset': offset_center,
                                                                       'fall_confidence': self.running_average_fall,
                                                                       'skipped': False})

    def handle_fall(self):
        """
        This function is responsible for handling a detected fall. It collects two frames and then calls
        a user-defined function (fall_handler_fn) in order to further process the data
        """
        images = []
        images.append(self.last_img)

        # Go towards the fall
        for i in range(self.distance_fall):
            self.safe_translate(2, lambda x: None)
            self.last_img = self.robot.get_camera_data()
            poses = self.pose_estimator.infer(self._get_infer_image())
            self.wait()
            if len(poses) > 0:
                self.last_pose = poses[0]
            else:
                break
            size = calculate_body_area(self.last_pose, self.image_width, self.image_height) * 2
            if size > self.min_distance_size:
                break
        images.append(self.last_img)

        self.fall_handler_fn(images)

    def safe_translate(self, distance_command, visualization_handler):
        """
        Allows for translating the robot after checking for obstacles
        @param distance_command: translation comamnd
        @param visualization_handler: command to use for the translation
        @return: True, if the command was executed, False, otherwise
        """

        if self.enable_depth_perception:
            from mxnet.gluon.data.vision import transforms
            import mxnet as mx
            import PIL.Image as pil

            img = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2RGB)
            img = pil.fromarray(np.uint8(img))
            img = img.resize((640, 192), pil.LANCZOS)
            img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=self.ctx)

            outputs = self.collision_model.predict(img)
            outputs = outputs[("disp", 0)]
            outputs = outputs.squeeze().as_in_context(mx.cpu()).asnumpy()

        if self.enable_depth_perception and np.mean(outputs) > self.collision_depth_threshold:
            print("Collision")
            return False
        else:
            self.robot.translate(distance_command, visualization_handler)
            return True

    def wait(self):
        time.sleep(self.infer_delay)
