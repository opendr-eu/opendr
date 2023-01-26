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
import tf
import tf2_ros
import tf.transformations

# ROS Messages
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

# Math Libraries
import numpy as np
from skimage.draw import line

from collections import defaultdict, deque

# Project Libraries
from map_simulator.utils import map2world, world2map, tf_frame_normalize


class GroundTruthMapping:
    """
    Class for generating a ground truth map from deterministic (noiseless) odometry and measurement messages,
    instead of running an entire SLAM stack.
    """

    def __init__(self):
        """
        Constructor
        """

        rospy.init_node('gt_mapping')

        self._tf_listener = tf.TransformListener()

        self._map_frame = tf_frame_normalize(rospy.get_param("~map_frame", "map"))
        self._ref_frame = tf_frame_normalize(rospy.get_param("~ref_frame", "map"))

        max_scan_buffer_len = rospy.get_param("~max_scan_buffer_len", 1000)
        self._occ_threshold = rospy.get_param("~occ_threshold", 0.25)

        self._standalone = rospy.get_param("~standalone", False)
        map_resolution = rospy.get_param("~resolution", 0.05)

        if self._standalone:
            self._map_resolution = map_resolution
        else:
            self._map_resolution = None

        self._scan_buffer = deque(maxlen=max_scan_buffer_len)

        self._map_hits = defaultdict(int)
        self._map_visits = defaultdict(int)
        self._width = 0
        self._height = 0
        self._min_x = None
        self._min_y = None
        self._max_x = None
        self._max_y = None
        self._map_origin = np.zeros(2)
        self._msg_seq = 0

        self._loc_only = False

        self._sub_map = rospy.Subscriber("/map", OccupancyGrid, self._map_callback)
        self._sub_eos = rospy.Subscriber("endOfSim", Bool, self._eos_callback)
        self._gen_map = rospy.Subscriber("genMap", Bool, self._gen_map_callback)
        self._sub_scan = rospy.Subscriber("/GT/base_scan", LaserScan, self._sensor_callback)
        self._sub_doLoc = rospy.Subscriber("doLocOnly", Bool, self._loc_only_callback)
        self._pub_map = rospy.Publisher("/GT/map", OccupancyGrid, queue_size=1)

        rospy.spin()

    def _loc_only_callback(self, msg):
        self._loc_only = msg.data

    def _sensor_callback(self, msg):
        """
        Function to be called each time a laser scan message is received.
        It computes the pose and endpoints of the laser beams and stores them in a queue,
        waiting for the right time to compute the map.

        :param msg: (sensor_msgs.LaserScan) Received Laser Scan message.

        :return: None
        """

        # Stop registering scans if no more mapping is taking place
        if self._loc_only:
            return

        try:
            # self._tf_listener.waitForTransform(self._ref_frame, msg.header.frame_id, msg.header.stamp,
            #                                    rospy.Duration(1))
            laser_pose, laser_orientation = self._tf_listener.lookupTransform(self._ref_frame, msg.header.frame_id,
                                                                              msg.header.stamp)

        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException, tf2_ros.TransformException) as e:
            rospy.logwarn("Couldn't find transform for scan {}. {}".format(msg.header.seq, e))
            return

        lp = np.array([laser_pose[0], laser_pose[1]])  # Laser Pose
        _, _, lp_th = tf.transformations.euler_from_quaternion(laser_orientation)  # Laser Orientation

        self._min_range = msg.range_min
        self._max_range = msg.range_max

        ranges = np.array(msg.ranges)
        max_range = ranges >= self._max_range
        ranges = np.clip(ranges, self._min_range, self._max_range)

        bearings = np.linspace(msg.angle_min, msg.angle_max, ranges.shape[0])
        bearings += lp_th

        cos_sin = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)
        endpoints = np.multiply(ranges.reshape((-1, 1)), cos_sin)
        endpoints += lp

        if self._map_resolution is None or self._standalone:
            # Store the scan data until we receive our first map message and thus know the map's resolution
            self._scan_buffer.append((lp, endpoints, max_range))
        else:
            self._register_scan(lp, endpoints, max_range)

        rospy.loginfo("Scan {} received and added to buffer at pose ({}): ({:.3f}, {:.3f}), theta: {:.3f}.".format(
            msg.header.seq, msg.header.frame_id, lp[0], lp[1], lp_th))

    def _min_max_cells(self, cell):
        c_x = cell[0]
        c_y = cell[1]

        if self._min_x is None:
            self._min_x = c_x
        elif c_x < self._min_x:
            self._min_x = c_x
        if self._max_x is None:
            self._max_x = c_x
        elif c_x > self._max_x:
            self._max_x = c_x

        if self._min_y is None:
            self._min_y = c_y
        elif c_y < self._min_y:
            self._min_y = c_y
        if self._max_y is None:
            self._max_y = c_y
        elif c_y > self._max_y:
            self._max_y = c_y

    def _register_scan(self, laser_pose, endpoints, max_range):
        ilp = world2map(laser_pose, np.zeros(2), self._map_resolution)
        i_endpoints = world2map(endpoints, np.zeros(2), self._map_resolution)

        for i, i_ep in enumerate(i_endpoints):

            line_rows, line_cols = line(ilp[0], ilp[1], i_ep[0], i_ep[1])
            line_indexes = np.array(zip(line_rows, line_cols))

            if not max_range[i]:
                occ_indexes = line_indexes[-1]
                # Increment hit cell
                hit_c = tuple(map2world(occ_indexes, np.zeros(2), self._map_resolution, rounded=True))
                self._map_hits[hit_c] += 1

            # Increment visited cells
            for visit in line_indexes:
                visit_c = tuple(map2world(visit, np.zeros(2), self._map_resolution, rounded=True))
                self._map_visits[visit_c] += 1
                if self._standalone:
                    self._min_max_cells(visit_c)

    def _update_map_params(self):
        if not self._standalone:
            return

        dims = world2map(np.array([self._max_x, self._max_y]), np.zeros(2), self._map_resolution) - \
            world2map(np.array([self._min_x, self._min_y]), np.zeros(2), self._map_resolution)

        self._height = dims[1] + 1
        self._width = dims[0] + 1
        self._map_origin = np.array([self._min_x, self._min_y])
        self._msg_seq += 1
        self._stamp = rospy.Time.now()

    def _gen_map_callback(self, msg):
        if not msg.data:
            return

        self._update_map()
        self._publish_map()

    def _eos_callback(self, msg):
        if not msg.data:
            return

        self._update_map()
        self._publish_map()

        rospy.signal_shutdown("Received EndOfSimulation. Shutting Down")

    def _map_callback(self, msg):
        """
        Function to be called each time a map message is received,
        just to copy the SLAM generated map properties for easier comparison.
        It:
            * takes the metadata from the published map (height, width, resolution, map origin),
            * creates a map if it is the first time,
            * checks if the map size changed in subsequent times and enlarges it in case it did,
            * takes all the poses and endpoints from the queue and updates the map values,
            * thresholds the map values,
            * publishes a ground truth occupancy map

        :param msg: (nav_msgs.OccupancyGrid) Received Map message.

        :return: None
        """

        # Set map attributes
        self._height = msg.info.height
        self._width = msg.info.width
        if self._map_resolution is None:
            self._map_resolution = msg.info.resolution

        if msg.info.resolution != self._map_resolution:
            raise ValueError("Map resolution changed from last time {}->{}. I can't work in these conditions!".format(
                self._map_resolution, msg.info.resolution))

        self._map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self._msg_seq = msg.header.seq
        self._stamp = msg.header.stamp

        self._update_map()
        self._publish_map()

    def _update_map(self):
        """
        Function for converting the endpoints to the center points of the grid cells, computing the cells crossed by
        the beams and mark those indexes as occ or free in the grid map, for every received scan still waiting
        to be processed in the buffer.

        :return: None
        """

        while self._scan_buffer:
            laser_pose, endpoints, max_range = self._scan_buffer.popleft()
            self._register_scan(laser_pose, endpoints, max_range)

        self._update_map_params()

    def _publish_map(self):

        # Compute Occupancy value as hits/visits from the default dicts
        map_shape = (self._width, self._height)
        occ_map = -1 * np.ones(map_shape, dtype=np.int8)

        # Freeze a snapshot (copy) of the current hits and visits in case a scan message comes in and alters the values.
        visits_snapshot = self._map_visits.copy()
        hits_snapshot = self._map_hits.copy()
        for pos, visits in visits_snapshot.iteritems():
            if visits <= 0:
                continue

            ix, iy = world2map(pos, self._map_origin, self._map_resolution)
            # Ignore cells not contained in image
            if ix >= self._width or iy >= self._height:
                continue

            hits = hits_snapshot[pos]
            tmp_occ = float(hits) / float(visits)
            if 0 <= tmp_occ < self._occ_threshold:
                occ_map[ix, iy] = 0
            elif tmp_occ >= self._occ_threshold:
                occ_map[ix, iy] = 100

        # Build Message and Publish
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = self._map_frame
        map_msg.header.stamp = self._stamp
        map_msg.header.seq = self._msg_seq
        map_msg.info.resolution = self._map_resolution
        map_msg.info.origin.position.x = self._map_origin[0]
        map_msg.info.origin.position.y = self._map_origin[1]
        map_msg.info.height = self._height
        map_msg.info.width = self._width
        map_msg.data = np.ravel(np.transpose(occ_map)).tolist()

        rospy.loginfo("Publishing map at {} with seq {}.".format(self._map_frame, self._msg_seq))

        self._pub_map.publish(map_msg)
