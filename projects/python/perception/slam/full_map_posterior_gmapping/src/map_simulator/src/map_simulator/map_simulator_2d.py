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

# ROS Libraries
import rospy
from tf.transformations import quaternion_from_euler
import rosbag

# ROS Messages
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

import os.path
from time import sleep, time
import sys

import simplejson as json

# Math Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from collections import deque

# Project Libraries
from map_simulator.geometry.primitives import Pose, Line
from map_simulator.map_obstacles import PolygonalObstacle
from map_simulator.robot_commands.move import MovePoseCommand, MoveInterpolationCommand, MoveLinearCommand, \
    MoveRotationCommand, MoveCircularCommand
from map_simulator.robot_commands.misc import ScanCommand, CommentCommand, SleepCommand
from map_simulator.robot_commands.message import BoolMessageCommand
from map_simulator.geometry.transform import rotate2d
from map_simulator.utils import tf_frame_normalize, tf_frame_join


class MapSimulator2D:
    """
    Class for simulating a robot's pose and it's scans using a laser sensor given a series of move commands around a map
    defined with polygonal obstacles.
    """

    def __init__(self, in_file, include_path, override_params=None):
        """
        Constructor

        :param in_file: (string) Name of the main robot parameter file. Actual file might include more files.
        :param include_path: (list) List of path strings to search for main and included files.
        :param override_params: (dict) Dictionary of parameter:value pairs to override any configuration
                                       defined in the files.
        """

        rospy.init_node('mapsim2d', anonymous=True)

        # Configuration keys that don't get overridden when importing json files and command line arguments,
        # but rather get appended to be able to combine multiple maps, commands, etc. from different sources.
        # Note: Parameter value must be of list type to be able to append to it!
        self.__cfg_append_keys = [
            "commands",
            "obstacles"
        ]

        # Default Settings Dictionary
        # Add extra (immutable) parameters with their default values and documentation here:
        # Other properties that will change during execution should have their own fields (e.g. pose).
        self.__param_defaults = {
            "map_frame":   {"def": "map",       "desc": "Map TF Frame"},
            "odom_frame":  {"def": "odom",      "desc": "Odometry TF Frame"},
            "base_frame":  {"def": "base_link", "desc": "Base TF Frame"},
            "laser_frame": {"def": "laser_link", "desc": "Laser TF Frame"},

            "gt_prefix":  {"def": "GT",  "desc": "Ground Truth TF prefix for the pose and measurement topics"},
            "odo_prefix": {"def": "odo", "desc": "Odometry TF prefix for the pose and measurement topics"},

            "base_to_laser_tf": {"def": [[0, 0], 0], "desc": "Base to Laser Transform"},

            "scan_topic": {"def": "base_scan", "desc": "ROS Topic for scan messages"},

            "deterministic": {"def": False,           "desc": "Deterministic process"},

            "move_noise_type": {"def": "odom", "desc": "Type of movement noise [linear|odom]"},
            "pose_sigma": {"def": [[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]], "desc": "Movement by Pose Covariance Matrix (3x3)"},

            "odometry_alpha": {"def": [0., 0., 0., 0.], "desc": "Movement by Odometry Covariance Matrix (3x3)"},

            "measurement_sigma": {"def": [[0., 0.],
                                          [0., 0.]],     "desc": "Measurement Covariance Matrix (2x2)"},

            "num_rays":  {"def": 180,         "desc": "Number of Sensor beams per scan"},
            "start_ray": {"def": - np.pi / 2, "desc": "Starting scan angle (in Radians)"},
            "end_ray":   {"def":   np.pi / 2, "desc": "End scan angle (in Radians)"},
            "max_range": {"def": 20,          "desc": "Laser Scanner Max. Range (in m)"},

            "meas_per_move": {"def": 1, "desc": "If >= 0: Number of measurements per move command.\\"
                                                "If < 0 : Number of moves before doing a scan."},

            "move_time_interval": {"def": 1000.0,
                                   "desc": "Time (in ms) that a movement command takes"},
            "scan_time_interval": {"def":   50.0,
                                   "desc": "Time (in ms) that a measurement command takes"},

            "render_move_pause": {"def": 0.5,
                                  "desc": "Time (in s) that the simulation pauses after each move action"},
            "render_sense_pause": {"def": 0.35,
                                   "desc": "Time (in s) that the simulation pauses after each sensing action"},
            "render_margin": {"def": 1.0,
                              "desc": "Margin (in m) to be left around the maximum bounding box of the simulation."}
        }

        # Command Classes and Callback functions for pre-processing and runtime execution
        # To add extra command types, create classes inheriting from one of the classes in robot_commands
        # and insert an entry  with its pre-processing and runtime callbacks here:
        cmd_callbacks = {
            # Movement Commands
            "pose":               {"class": MovePoseCommand,          "callback": self._callback_cmd_moves},
            "odom":               {"class": MovePoseCommand,          "callback": self._callback_cmd_moves},
            "interpolation":      {"class": MoveInterpolationCommand, "callback": self._callback_cmd_moves},
            "linear":             {"class": MoveLinearCommand,        "callback": self._callback_cmd_moves},
            "rotation":           {"class": MoveRotationCommand,      "callback": self._callback_cmd_moves},
            "circular":           {"class": MoveCircularCommand,      "callback": self._callback_cmd_moves},

            # Measurement Commands
            "scan":               {"class": ScanCommand,              "callback": self._callback_cmd_scans},

            # Message Commands
            "bool_msg":           {"class": BoolMessageCommand,       "callback": self._callback_cmd_msg},

            # Misc Commands
            "sleep":              {"class": SleepCommand,             "callback": self._callback_cmd_sleep},
            "comment":            {"class": CommentCommand,           "callback": self._callback_cmd_comment}
        }

        # Obstacle class dictionary
        # To add extra obstacle types, create classes inheriting from map_simulator.map_obstacles.obstacle.Obstacle
        obstacle_classes = {
            "polygon": PolygonalObstacle
            # TODO: Add more obstacle types E.G.:
            #   "ellipse": EllipticalrObstacle,
            #   "bezier": BezierObstacle,
        }

        # Main Configuration lists/dicts
        self._obstacles = []
        self._params = {}
        self._commands = []

        # ROS Bag
        self._bag = None

        # Preview Simulation
        self._do_display = False

        # Simulation Axes object
        self._axes = None

        # Preview bounds
        self._min_x = 0
        self._min_y = 0
        self._max_x = 0
        self._max_y = 0

        # Counters
        # Move counter
        self._move_cnt = 0
        # Message counters
        self._laser_msg_seq = 0
        self._tf_msg_seq = 0

        self._noisy_pose = Pose(np.zeros(2), np.zeros(1))
        self._real_pose = Pose(np.zeros(2), np.zeros(1))
        self._real_sensor_pose = Pose(np.zeros(2), np.zeros(1))

        config = self._import_json(in_file, include_path)

        if override_params is not None:
            override_params = json.loads(override_params)
            self._update_param_dict(config, override_params)

        # Parse Map Obstacles
        try:
            config_obstacles = config['obstacles']

        except KeyError:
            rospy.logwarn("No obstacles defined in map file")
            config_obstacles = []

        for obs_dict in config_obstacles:
            try:
                obstacle = obstacle_classes[obs_dict['type']](obs_dict)
                min_x, min_y, max_x, max_y = obstacle.get_bounding_box()
                self._update_preview_bounds(min_x, min_y)
                self._update_preview_bounds(max_x, max_y)
                self._obstacles.append(obstacle)
            except (KeyError, AttributeError) as e:
                rospy.logwarn("Unable to parse obstacle {}, ignoring.\n {}".format(obs_dict, e))

        # Parse Parameters
        self._set_dict_param(config)

        # Parse Uncertainty Parameters
        self._params['pose_sigma'] = np.array(self._params['pose_sigma'])
        self._params['odometry_alpha'] = np.array(self._params['odometry_alpha'])
        self._params['measurement_sigma'] = np.array(self._params['measurement_sigma'])

        # Parse Timestamp parameters
        try:
            self._current_time = rospy.Time(int(config['initial_timestamp']))
        except (KeyError, ValueError, TypeError) as e:
            rospy.logwarn("Unable to set start time. Using current system time as initial time stamp.\n %s", e)
            self._current_time = rospy.Time.from_sec(time())

        # Parse Start Pose
        noisy_position = np.zeros(2)
        try:
            noisy_position = np.array(config['start_position'])
        except KeyError:
            rospy.logwarn("No initial position defined in config file. Starting at (0, 0)")

        noisy_orientation = np.zeros(1)
        try:
            noisy_orientation = np.array(config['start_orientation'])
        except KeyError:
            rospy.logwarn("No initial orientation defined in config file. Starting with theta=0")

        self._noisy_pose = Pose(noisy_position, noisy_orientation)
        self._real_pose = Pose(np.copy(noisy_position), np.copy(noisy_orientation))

        # Initialize sensor position from starting pose
        self._compute_sensor_pose()

        # Create own command list and initialize with initial pose.
        pose0_cmd_dict = {"position": self._real_pose.position, "orientation": self._real_pose.orientation}
        pose0_cmd_cfg = cmd_callbacks['pose']
        pose0_cmd_cls = pose0_cmd_cfg['class']
        pose0_cmd_cb = pose0_cmd_cfg['callback']
        self._commands.append(pose0_cmd_cls(pose0_cmd_dict, pose0_cmd_cb, None))

        # Parse Commands
        conf_commands = []
        try:
            conf_commands = config['commands']
        except KeyError:
            rospy.logwarn("No commands defined in config file. Considering only starting position")

        last_pose = self._real_pose
        for cmd_cfg in conf_commands:
            try:
                cmd_config = cmd_callbacks[cmd_cfg['type']]
                cmd_class = cmd_config['class']
                cmd_callback = cmd_config['callback']

                parsed_cmd = cmd_class(cmd_cfg, cmd_callback, last_pose)
                self._commands.append(parsed_cmd)

            except KeyError as e:
                rospy.logwarn("Unknown command {}, ignoring.\nException {}".format(cmd_cfg, e.message))
                continue
            except AttributeError as e:
                rospy.logerr("Unable to parse command {}.Terminating!".format(cmd_cfg))
                raise e

            for pose in parsed_cmd.get_poses():
                last_pose = pose

                # Take into account robot moves for preview display box
                self._update_preview_bounds(last_pose.position[0], last_pose.position[1])

        # Add a margin of either the max range of the sensor (too large) or just 1m
        if self._params['render_margin'] is None:
            margin = self._params["max_range"] + 1
        else:
            margin = self._params['render_margin']

        self._min_x -= margin
        self._min_y -= margin
        self._max_x += margin
        self._max_y += margin

    def _set_dict_param(self, in_dict):
        """
        Method for setting a parameter's value, or take the default one if not provided.

        :param in_dict: (dict) Input dictionary from which (valid) parameters will be read.

        :return: None
        """

        for param in self.__param_defaults.keys():
            if param in in_dict:
                self._params[param] = in_dict[param]
            else:

                default = self.__param_defaults[param]['def']
                desc = self.__param_defaults[param]['desc']
                rospy.logwarn("Param {} undefined in config file. Using default value: {}.\nHelp: {}.".format(param,
                                                                                                              default,
                                                                                                              desc))
                self._params[param] = default

    def _update_param_dict(self, dest_dict, ori_dict):
        """
        Appends or overrides to a given destination configuration dictionary from an origin one.
        Append actions are configured by self.__cfg_append_keys in __init__.

        :param dest_dict: (dict) Destination dictionary where, if a parameter exists, entries will be overridden or
                                 appended to.
        :param ori_dict: (dict) Origin dictionary from where, if a parameter exists, entries will be copied from.
                                So that the update doesn't end up overriding anyways, the entry is deleted from
                                the origin dictionary if it is among the append keys.

        :return: None
        """

        # First append to select parameters
        for param in self.__cfg_append_keys:
            if param in dest_dict and param in ori_dict:
                dest_dict[param].extend(ori_dict[param])
                del ori_dict[param]

        # Then override remaining parameters
        dest_dict.update(ori_dict)

    def _build_include_queue(self, in_file, include_paths=None, include_queue=None, included=None):
        """
        Builds the include file queue to be able to parse the configuration files in desired order,
        i.e. in Depth-First order.
        It ignores files included more than once to avoid infinite loops.

        :param in_file: (string) Path string pointing to a JSON configuration file.
        :param include_paths: (list) List of paths where to search for files.
        :param include_queue: (deque) Queue of files to be included as read from the file's include statement.
        :param included: (set) Set of already included files to avoid duplication and infinite loops.

        :return: (deque) Queue of files to be included as read from the file's include statement.
        """

        if include_paths is None:
            include_paths = []
        if include_queue is None:
            include_queue = deque()
        if included is None:
            included = set()

        import_successful = False

        # Try to find file in all search paths
        for path in include_paths:
            in_path = os.path.join(path, in_file)
            in_path = os.path.expandvars(in_path)
            in_path = os.path.normpath(in_path)
            in_path = os.path.abspath(in_path)

            if in_path in included:
                rospy.logwarn("File %s already included. Ignoring to avoid infinite loops", in_file)
                import_successful = True
                break

            # If file is found in one of the paths, try to open it
            if os.path.isfile(in_path):
                try:
                    with open(in_path, "r") as f:
                        text = f.read()
                        file_config = json.loads(text)

                except IOError as e:
                    rospy.logwarn("Couldn't open {}.".format(in_path))
                    raise e
                except json.JSONDecodeError as e:
                    rospy.logerr("Couldn't parse {}.".format(in_path))
                    raise e
                else:
                    included.add(in_path)
                    import_successful = True

                    if "include" in file_config:
                        include_list = file_config["include"]
                        for include_subfile in include_list:
                            self._build_include_queue(include_subfile, include_paths, include_queue, included)
                        del file_config["include"]

                    include_queue.append(file_config)
                    rospy.loginfo("Added configuration of file %s to import queue", in_path)

                    break

        if not import_successful:
            rospy.logerr("Couldn't find {} in any of the search paths: [{}]".format(in_file, ", ".join(include_paths)))
            raise IOError("File {} could not be imported.".format(in_file))

        return include_queue

    def _import_json(self, in_file, include_paths):
        """
        Recursively import a JSON file and it's included subfiles.

        :param in_file: (string) Path string pointing to a JSON configuration file.
        :param include_paths: (list) List of paths where to search for files.

        :return: (dict) Dictionary of settings read from the JSON file(s).
        """

        include_queue = self._build_include_queue(in_file, include_paths)

        config = {}

        while include_queue:
            file_config = include_queue.popleft()
            self._update_param_dict(config, file_config)

        return config

    def simulate(self, output_file, display=False):
        """
        Main method for simulating and either saving the data to a ROSBag file, displaying it or both.

        :param output_file: (string|None) Path and filename to where the ROSBag file is to be saved.
                                          If None, then nothing will be saved to file, only simulated.
        :param display: (bool) Display the simulation. Set to False to run faster
                               (no pauses in the execution for visualization)

        :return: None
        """

        if output_file is None and not display:
            return

        self._do_display = display

        if output_file is not None:
            try:
                out_path = os.path.expanduser(output_file)
                out_path = os.path.expandvars(out_path)
                self._bag = rosbag.Bag(out_path, "w")

            except (IOError, ValueError) as e:
                rospy.logerr("Couldn't open %s", output_file)
                raise e

            rospy.loginfo("Writing ROS-Bag to : %s", out_path)

        if self._do_display:
            plt.ion()
            figure1 = plt.figure(1)
            self._axes = figure1.add_subplot(111)
            figure1.show()

        self._add_tf_messages()

        for p_cnt, cmd in enumerate(self._commands):

            # Execute the command (runs the command's callback function)
            cmd()

        rospy.loginfo("Finished simulation")
        if self._bag is not None:
            self._bag.close()
            rospy.loginfo("ROS-Bag saved and closed")

    def _callback_cmd_moves(self, cmd):
        """
        Runtime execution callback of move commands.
        It performs measurements depending on the "meas_num" parameter. If meas_num is non-negative, then it will
        execute <meas_scan> scans for every movement to a new pose. If meas_num is negative, then it will only perform
        scan once abs(<meas_scan>) movements have happened.

        :param cmd: (MoveCommand) Move Command object with get_poses() method.

        :return: None
        """

        meas_num = cmd.get_meas_per_pose()
        if meas_num is None:
            meas_num = int(self._params['meas_per_move'])

        det = cmd.get_deterministic()

        for pose in cmd.get_poses():

            # If multiple scans per move:
            if meas_num >= 0:
                # For each scan while stopped
                for _ in range(meas_num):
                    self._scan(deterministic=det)

            # If one scan per several moves:
            else:
                if self._move_cnt % abs(meas_num) == 0:
                    self._move_cnt = 0
                    self._scan(deterministic=det)

            # Move robot to new pose(s)
            self._move(pose, deterministic=det)
            self._move_cnt += 1

    def _callback_cmd_scans(self, cmd):
        """
        Runtime execution of scan command.
        It performs one or more scans of the environment, depending on the property scans of the command.

        :param cmd: (ScanCommand) Scan command with scans property.

        :return:
        """

        det = cmd.get_deterministic()

        for _ in range(cmd.scans):
            self._scan(deterministic=det)

    def _callback_cmd_sleep(self, cmd):
        """
        Runtime execution callback of sleep command.
        Increments the simulation clock by a given amount.

        :param cmd: (SleepCommand) Sleep command with property ms (sleep duration in ms).

        :return: None
        """

        self._increment_time(cmd.ms)

    def _callback_cmd_msg(self, cmd):
        """
        Runtime execution callback of a message command.
        Sends a boolean message.

        :param cmd: (MessageCommand) Message command with topic, msg, desc and do_print properties.

        :return: None
        """

        if self._bag is None:
            return

        self._bag.write(cmd.topic, cmd.msg, self._current_time)
        self._bag.flush()

        if cmd.do_print:
            rospy.loginfo("MSG: " + cmd.desc)

    @staticmethod
    def _callback_cmd_comment(cmd):
        """
        Processes a comment type command.

        :param cmd: (CommentCommand) Comment command with properties print and msg.

        :return: None
        """

        if cmd.do_print:
            rospy.loginfo("# " + cmd.msg)

    def _move(self, pose, deterministic=None):
        """
        Take a real pose and add noise to it. Then recompute the laser sensor pose and increment the time.
        If the parameter "deterministic" is set to True, then both the real and noisy poses will be equal.
        Otherwise, tne noisy pose will be computed depending on the "move_noise_type" parameter, which can be:
            * "odom": The movement is modeled as an initial rotation, then a translation and finally another rotation.
                      Noise is added to each of these steps as a weighted sum of the actual displacements with weights
                      defined by "odometry_alpha": [a1, a2, a3, a4].
            * "linear": Zero-mean gaussian noise with covariance matrix
                        "pose_sigma": [[sxx, sxy, sxth], [syx, syy, syth], [sthx, sthy, sthth]]
                        is added to the target pose.

        :param pose: (Pose) Next pose of the robot defined as [[x, y], th]

        :return: None
        """

        target_position = pose.position
        target_orientation = pose.orientation

        if deterministic is None:
            deterministic = self._params['deterministic'] or self._tf_msg_seq == 0

        if deterministic:
            new_noisy_position = target_position
            new_noisy_orientation = target_orientation

        else:
            # Rotation/Translation/Rotation error
            if self._params['move_noise_type'] == "odom":

                # Compute delta in initial rotation, translation, final rotation
                delta_trans = target_position - self._real_pose.position
                delta_rot1 = np.arctan2(delta_trans[1], delta_trans[0]) - self._real_pose.orientation
                delta_rot2 = target_orientation - self._real_pose.orientation - delta_rot1
                delta_trans = np.matmul(delta_trans, delta_trans)
                delta_trans = np.sqrt(delta_trans)

                delta_rot1_hat = delta_rot1
                delta_trans_hat = delta_trans
                delta_rot2_hat = delta_rot2

                # Add Noise
                alpha = self._params['odometry_alpha']
                delta_rot1_hat += np.random.normal(0, alpha[0] * np.abs(delta_rot1) + alpha[1] * delta_trans)
                delta_trans_hat += np.random.normal(0, alpha[2] * delta_trans + alpha[3] * (np.abs(delta_rot1) +
                                                                                            np.abs(delta_rot2)))
                delta_rot2_hat += np.random.normal(0, alpha[0] * np.abs(delta_rot2) + alpha[1] * delta_trans)

                theta1 = self._noisy_pose.orientation + delta_rot1_hat
                new_noisy_position = self._noisy_pose.position + (delta_trans_hat *
                                                                  np.array([np.cos(theta1), np.sin(theta1)]).flatten())
                new_noisy_orientation = theta1 + delta_rot2_hat

            # Linear error
            else:

                noise = np.random.multivariate_normal(np.zeros(3), self._params['pose_sigma'])

                new_noisy_position = np.array(target_position + noise[0:1]).flatten()
                new_noisy_orientation = np.array(target_orientation + noise[2]).flatten()

        self._noisy_pose = Pose(new_noisy_position, new_noisy_orientation)
        self._real_pose = Pose(target_position, target_orientation)

        # Recompute sensor pose from new robot pose
        self._compute_sensor_pose()
        self._increment_time(self._params['move_time_interval'])
        self._tf_msg_seq += 1

        if self._do_display and (self._axes is not None):
            move_pause = float(self._params['render_move_pause'])
            self._render(pause=move_pause)

    def _scan(self, deterministic=None):
        """
        Generate a scan using ray tracing, add noise if configured and display it.

        :return: (np.ndarray, np.ndarray) Arrays of Measurements and Noisy Measurements respectively to be
                                          displayed and published.
                                          Each 2D array is comprised of [bearings, ranges].
        """

        meas, endpoints, hits = self._ray_trace()

        if deterministic is None:
            deterministic = self._params['deterministic']

        if deterministic:
            noisy_meas = meas
            meas_noise = np.zeros(2)
        else:
            meas_noise = np.random.multivariate_normal(np.zeros(2), self._params['measurement_sigma'],
                                                       size=meas.shape[0])
            noisy_meas = meas + meas_noise

        if self._do_display and (self._axes is not None):
            if self._params['deterministic']:
                noisy_endpoints = endpoints
            else:
                range_noises = meas_noise[:, 1]
                bearing_noises = meas_noise[:, 0]
                bearing_noises = np.array([np.cos(bearing_noises), np.sin(bearing_noises)])
                endpoint_noise = range_noises * bearing_noises
                endpoint_noise = endpoint_noise.transpose()
                noisy_endpoints = endpoints + endpoint_noise

            meas_pause = float(self._params['render_sense_pause'])
            meas_off_pause = float(self._params['render_move_pause'] - meas_pause)
            self._render(noisy_endpoints, hits, pause=meas_pause)
            self._render(pause=meas_off_pause)

        # Increment time and sequence
        self._increment_time(self._params['scan_time_interval'])
        self._laser_msg_seq += 1

        self._add_scan_messages(noisy_meas, det_measurements=meas)
        # Publish pose after each measurement, otherwise gmapping doesn't process the scans
        self._add_tf_messages()

        return meas, noisy_meas

    def _compute_sensor_pose(self):
        """
        Computes the real sensor pose from the real robot pose and the base_to_laser_tf transform.

        :return: None
        """
        tf_trans = np.array(self._params['base_to_laser_tf'][0])
        tf_rot = np.array(self._params['base_to_laser_tf'][1])

        rotation = rotate2d(self._real_pose.orientation)
        translation = rotation.dot(tf_trans)

        self._real_sensor_pose = Pose(self._real_pose.position + translation, self._real_pose.orientation + tf_rot)

    def _ray_trace(self):
        """
        Generates a set of laser measurements starting from the laser sensor pose until the beams either hit an obstacle
        or reach the maximum range.

        :return: (tuple) Tuple comprised of:
                             * bearing_ranges: ndarray of measurement angles
                             * endpoints: ndarray of (x,y) points where the laser hit or reached max_range,
                             * hits: ndarray of boolean values stating whether the beam hit an obstacle.
                                     True for hit, False for max_range measurement.
        """

        bearing_ranges = []
        endpoints = []
        hits = []

        bearing = self._params['start_ray']

        num_rays = self._params['num_rays']
        if num_rays > 1:
            num_rays -= 1

        bearing_increment = (self._params['end_ray'] - self._params['start_ray']) / num_rays

        for i in range(int(self._params['num_rays'])):

            theta = self._real_sensor_pose.orientation + bearing
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([c, s]).flatten()
            max_ray_endpt = self._real_sensor_pose.position + self._params['max_range'] * rotation_matrix
            ray = Line(self._real_sensor_pose.position, max_ray_endpt)

            # Add 1m to make sure that everyone knows that this was a max range measurement in case
            # no intersection is found
            min_range = self._params['max_range'] + 1
            min_endpt = max_ray_endpt
            hit = False

            for obstacle in self._obstacles:
                intersect = obstacle.intersection_with_line(ray)

                if intersect is not None:
                    beam = Line(ray.p1, intersect)
                    meas_range = beam.len
                    if min_range is None or meas_range < min_range:
                        min_range = meas_range
                        min_endpt = intersect
                        hit = True

            bearing_ranges.append([bearing, min_range])
            endpoints.append(min_endpt)
            hits.append(hit)

            bearing += bearing_increment

        bearing_ranges = np.array(bearing_ranges)
        endpoints = np.array(endpoints)
        hits = np.array(hits)

        return bearing_ranges, endpoints, hits

    @staticmethod
    def __add_transform(msg, ts, seq, p_frame, c_frame, position, rotation):
        """
        Function for appending a transform to an existing TFMessage.

        :param msg: (tf2_msgs.TFMessage) A TF message to append a transform to.
        :param ts: (rospy.Time) A time stamp for the Transform's header.
        :param seq: (int) The number/sequence of the TF message.
        :param p_frame: (string) Parent frame of the transform.
        :param c_frame: (string) Child frame of the transform.
        :param position: (geometry_msg.Point) A point representing the translation between frames.
        :param rotation: (list) A quaternion representing the rotation between frames.

        :return: None
        """

        tran = TransformStamped()

        tran.header.stamp = ts
        tran.header.seq = seq
        tran.header.frame_id = p_frame
        tran.child_frame_id = c_frame

        tran.transform.translation = position
        tran.transform.rotation.x = rotation[0]
        tran.transform.rotation.y = rotation[1]
        tran.transform.rotation.z = rotation[2]
        tran.transform.rotation.w = rotation[3]

        msg.transforms.append(tran)

    def __add_tf_msg(self, msg, real_pose=False, tf_prefix="", update_laser_tf=True, publish_map_odom=False):
        """
        Appends the tf transforms to the passed message with the real/noisy pose of the robot
        and (optionally) the laser sensor pose.

        :param msg: (tf2_msgs.TFMessage) A TF message to append all the transforms to.
        :param real_pose: (bool)[Default: False] Publish real pose if True, noisy pose if False.
        :param tf_prefix: (string)[Default: ""] Prefix to be prepended to each TF Frame
        :param update_laser_tf: (bool)[Default: True] Publish base_link->laser_link tf if True.
        :param publish_map_odom: (bool)[Default: False] Publish the map->odom tf transform if True.

        :return: None
        """

        ts = self._current_time
        seq = self._tf_msg_seq

        odom_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['odom_frame'])))
        base_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['base_frame'])))

        if publish_map_odom:
            map_frame = str(self._params['map_frame'])
            zero_pos = Point(0.0, 0.0, 0.0)
            zero_rot = quaternion_from_euler(0.0, 0.0, 0.0)

            self.__add_transform(msg, ts, seq, map_frame, odom_frame, zero_pos, zero_rot)

        if real_pose:
            pos_x = float(self._real_pose.position[0])
            pos_y = float(self._real_pose.position[1])
            theta = float(self._real_pose.orientation)

        else:
            pos_x = float(self._noisy_pose.position[0])
            pos_y = float(self._noisy_pose.position[1])
            theta = float(self._noisy_pose.orientation)

        odom_pos = Point(pos_x, pos_y, 0.0)
        odom_rot = quaternion_from_euler(0.0, 0.0, theta)

        self.__add_transform(msg, ts, seq, odom_frame, base_frame, odom_pos, odom_rot)

        if update_laser_tf:
            laser_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['laser_frame'])))

            lp_x = float(self._params['base_to_laser_tf'][0][0])
            lp_y = float(self._params['base_to_laser_tf'][0][1])
            lp_th = float(self._params['base_to_laser_tf'][1][0])

            laser_pos = Point(lp_x, lp_y, 0.0)
            laser_rot = quaternion_from_euler(0.0, 0.0, lp_th)

            self.__add_transform(msg, ts, seq, base_frame, laser_frame, laser_pos, laser_rot)

    def _add_tf_messages(self, add_gt=True, add_odom=True, update_laser_tf=True):
        """
        Function for adding all TF messages (Noisy pose, GT pose, Odom pose) at once.

        :param add_gt: (bool)[Default: True] Publish the ground truth transforms.
        :param add_odom (bool)[Default: True] Publish the plain odometry transforms.
        :param update_laser_tf: (bool)[Default: True] Publish base_link->laser_link tf if True.

        :return: None
        """

        if self._bag is None:
            return

        tf2_msg = TFMessage()

        # Noisy Pose
        first_msg = self._tf_msg_seq == 0
        self.__add_tf_msg(tf2_msg, real_pose=False, tf_prefix="", update_laser_tf=update_laser_tf,
                          publish_map_odom=first_msg)

        # Ground Truth Pose
        if add_gt:
            gt_prefix = str(self._params['gt_prefix'])
            self.__add_tf_msg(tf2_msg, real_pose=True, tf_prefix=gt_prefix, update_laser_tf=update_laser_tf,
                              publish_map_odom=True)

        # Odometry Pose
        if add_odom:
            odo_prefix = str(self._params['odo_prefix'])
            self.__add_tf_msg(tf2_msg, real_pose=False, tf_prefix=odo_prefix, update_laser_tf=update_laser_tf,
                              publish_map_odom=True)

        self._bag.write("/tf", tf2_msg, self._current_time)
        self._bag.flush()

    def __add_scan_msg(self, topic, frame, measurements):
        """
        Publish a LaserScan message to a ROSBag file with the given measurement ranges.

        :param measurements: (ndarray) 2D Array of measurements to be published.
                                       Measured ranges must be in measurements[:, 1]

        :return: None
        """

        if self._bag is None:
            return

        meas_msg = LaserScan()

        meas_msg.header.frame_id = frame  # topic_prefix + self._params['laser_frame']
        meas_msg.header.stamp = self._current_time
        meas_msg.header.seq = self._laser_msg_seq

        meas_msg.angle_min = self._params['start_ray']
        meas_msg.angle_max = self._params['end_ray']
        if self._params['num_rays'] > 1:
            meas_msg.angle_increment = (meas_msg.angle_max - meas_msg.angle_min) / (self._params['num_rays'] - 1)
        else:
            meas_msg.angle_increment = 0.0

        meas_msg.range_min = 0.0
        meas_msg.range_max = self._params['max_range']

        meas_msg.ranges = measurements[:, 1]

        self._bag.write(topic, meas_msg, self._current_time)
        self._bag.flush()

    def _add_scan_messages(self, noisy_measurements, det_measurements=None, add_gt=True, add_odom=True):
        """
        Function for adding all scan messages (Noisy pose, GT pose, Odom pose) at once.

        :param noisy_measurements: (ndarray) 2D Array of noisy measurements to be published.
                                             Measured ranges must be in measurements[:, 1]
        :param det_measurements: (ndarray)[Default: None] 2D Array of deterministic measurements to be published.
                                                          Measured ranges must be in measurements[:, 1].
                                                          If None, then ground truth won't be published,
                                                          even if add_gt is True.
        :param add_gt: (bool)[Default: True] Add scan message from ground truth pose frame if True
                                             and det_measurements is not None.
        :param add_odom: (bool)[Default: True] Add scan messages from odom pose frame if True.

        :return: None
        """

        if self._bag is None:
            return

        topic = str(self._params['scan_topic'])
        frame = str(self._params['laser_frame'])

        self.__add_scan_msg(topic, frame, noisy_measurements)

        if add_gt and det_measurements is not None:
            gt_prefix = str(self._params['gt_prefix'])
            gt_topic = '/' + tf_frame_normalize(tf_frame_join(gt_prefix, topic))
            gt_frame = tf_frame_normalize(tf_frame_join(gt_prefix, frame))

            self.__add_scan_msg(gt_topic, gt_frame, det_measurements)

        if add_odom:
            odo_prefix = str(self._params['odo_prefix'])
            odo_topic = '/' + tf_frame_normalize(tf_frame_join(odo_prefix, topic))
            odo_frame = tf_frame_normalize(tf_frame_join(odo_prefix, frame))

            self.__add_scan_msg(odo_topic, odo_frame, noisy_measurements)

    def _increment_time(self, ms):
        """
        Increment the internal simulated time variable by a given amount.

        :param ms: (float) Time in miliseconds to increment the time.

        :return: None
        """

        secs = ms
        nsecs = secs
        secs = int(secs / 1000)
        nsecs = int((nsecs - 1000 * secs) * 1e6)

        self._current_time += rospy.Duration(secs, nsecs)

    def _update_preview_bounds(self, x, y):
        """
        Updates the preview boundaries by checking if a point is smaller or greater than the current preview
        window boundaries.

        :param x: (float) Point's x coordinate to be included in the preview window.
        :param y: (float) Point's y coordinate to be included in the preview window.

        :return: None
        """

        self._min_x = min(self._min_x, x)
        self._min_y = min(self._min_y, y)
        self._max_x = max(self._max_x, x)
        self._max_y = max(self._max_y, y)

    def _draw_map(self):
        """
        Draw the map's obstacles.

        :return: None
        """

        for obstacle in self._obstacles:
            obstacle.draw(self._axes)

    def _draw_robot(self, real=False):
        """
        Draw the robot's real/noisy pose.

        :param real: (bool) Draw the real pose if True, the noisy one if False.

        :return: None
        """
        robot_size = 0.05

        if real:
            pos_x = self._real_pose.position[0]
            pos_y = self._real_pose.position[1]
            theta = self._real_pose.orientation
            orientation_inipt = self._real_pose.position
            robot_color = "tab:green"
            z_order = 4
        else:
            pos_x = self._noisy_pose.position[0]
            pos_y = self._noisy_pose.position[1]
            theta = self._noisy_pose.orientation
            orientation_inipt = self._noisy_pose.position
            robot_color = "tab:purple"
            z_order = 2

        robot_base = plt.Circle((pos_x, pos_y), robot_size, color=robot_color, zorder=z_order)
        self._axes.add_artist(robot_base)

        orientation_endpt = robot_size * np.array([np.cos(theta), np.sin(theta)]).reshape((2,))
        orientation_endpt += orientation_inipt
        orientation_line = np.stack((orientation_inipt, orientation_endpt)).transpose()

        robot_orientation = plt.Line2D(orientation_line[0], orientation_line[1], color='white', zorder=z_order + 1)
        self._axes.add_artist(robot_orientation)

        return robot_base

    def _draw_beams(self, beams, hits):
        """
        Draw each of the laser measurement beams from the robot's real pose.

        :param beams: (ndarray|None) List of the beams' endpoints (x, y). None to not display the measurements.
        :param hits: (ndarray) List of booleans stating for each beam whether it hit an obstacle (True)
                               or reached max_range (False).

        :return: None
        """

        if beams is None:
            return

        for i, beam in enumerate(beams):
            ray = np.array([self._real_sensor_pose.position, beam])
            ray = ray.transpose()

            if hits[i]:
                self._axes.plot(ray[0], ray[1], 'tab:red', marker='.', linewidth=0.7)
            else:
                self._axes.plot(ray[0], ray[1], 'tab:red', marker='1', dashes=[10, 6], linewidth=0.5)

    def _render(self, beam_endpoints=None, hits=None, pause=0.25):
        """
        Renders a graphical view of the map, the current state of the robot and the measurements using Matplotlib

        :param beam_endpoints: Numpy array of the laser endpoints. None if measurements are not to be displayed.
        :param hits: Numpy array of booleans indicating whether each laser beam actually hit an obstacle or was a
        max. range reading.

        :return: None
        """

        self._axes.clear()

        self._axes.set_aspect('equal', 'box')
        self._axes.set_xlim([self._min_x, self._max_x])
        self._axes.set_xbound(self._min_x, self._max_x)
        self._axes.set_ylim(self._min_y, self._max_y)
        self._axes.set_ybound(self._min_y, self._max_y)

        self._draw_map()
        self._draw_beams(beam_endpoints, hits)
        noisy_robot_handle = self._draw_robot()
        real_robot_handle = self._draw_robot(real=True)

        self._axes.xaxis.set_major_locator(MultipleLocator(1))
        self._axes.yaxis.set_major_locator(MultipleLocator(1))
        self._axes.xaxis.set_minor_locator(AutoMinorLocator(4))
        self._axes.yaxis.set_minor_locator(AutoMinorLocator(4))
        self._axes.grid(which='major', color='#CCCCCC')
        self._axes.grid(which='minor', color='#CCCCCC', linestyle=':')

        self._axes.legend((real_robot_handle, noisy_robot_handle), ("Real Pose", "Noisy Odometry"), loc='lower center')

        self._axes.grid(True)

        plt.draw()
        plt.pause(0.0001)

        sleep(pause)

    @staticmethod
    def _print_status(percent, length=40):
        """
        Prints the percentage status of the simulation.

        :param percent: (float) The percentage to be displayed numerically and in the progress bar.
        :param length: (int)[Default: 40] Length in characters that the progress bar will measure.

        :return: None
        """

        # Erase line and move to the beginning
        sys.stdout.write('\x1B[2K')
        sys.stdout.write('\x1B[0E')

        progress = "Simulation Progress: ["

        for i in range(0, length):
            if i < length * percent:
                progress += '#'
            else:
                progress += ' '
        progress += "] " + str(round(percent * 100.0, 2)) + "%"

        sys.stdout.write(progress)
        sys.stdout.flush()
