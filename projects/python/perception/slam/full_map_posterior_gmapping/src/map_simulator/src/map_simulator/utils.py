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

import numpy as np
import errno
import os


def mkdir_p(path):

    if os.path.exists(path):
        return

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >= 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def map_msg_to_numpy(msg, dtype=np.float64):
    """
    Reshapes a map's data from a 1D list to a 2D ndarray.

    :param msg: (nav_msgs.OccupancyMap|gmapping.doubleMap) A map message.
    :param dtype: (numpy type)[Default: np.float64] Type of the array to be returned.

    :return: (numpy ndarray) The map, reshaped as a 2D matrix.
    """
    w = msg.info.width
    h = msg.info.height

    reshaped_map = np.array(msg.data)
    reshaped_map = reshaped_map.reshape(h, w)
    reshaped_map = np.flipud(reshaped_map)
    reshaped_map = reshaped_map.astype(dtype)

    return reshaped_map


def map_msg_extent(msg):
    """
    Returns the extent of the map in world coordinates

    :param msg: ((nav_msgs.OccupancyMap|gmapping.doubleMap) A map message.

    :return: (list) The extents of the map in world coordinates [x0, x1, y0, y1]
    """

    w = msg.info.width
    h = msg.info.height

    # Set the plot's extension in world coordinates for meaningful plot ticks
    delta = msg.info.resolution
    x0 = msg.info.origin.position.x
    y0 = msg.info.origin.position.y
    x1 = x0 + w * delta
    y1 = y0 + h * delta

    extent = [x0, x1, y0, y1]

    return extent


def tf_frame_split(tf_frame):
    """
    Function for splitting a frame into its path components for easier comparison, ignoring slashes ('/').

    :param tf_frame: (string) TF frame

    :return: (list) List of TF Frame path components.
                    E.g.: for '/GT/base_link' it returns ['GT', 'base_link']
    """

    return filter(None, tf_frame.split('/'))


def tf_frame_join(*args):
    """
    Function for joining a frame list into a string path. Opposite to tf_frame_split.

    :param args: (string|list) Strings or List of strings for path components to be joined into a single TF Frame.

    :return: (string) A fully formed TF frame
    """

    tf_path = ''

    for arg in args:
        if isinstance(arg, list):
            tf_path += '/' + '/'.join(arg)
        elif isinstance(arg, str):
            tf_path += '/' + arg

    return tf_path[1:]


def tf_frame_normalize(tf_frame):
    """
    Function for normalizing a TF frame string.

    :param tf_frame: (string) String of a single TF Frame.

    :return: (string) A standardized TF frame
    """

    return tf_frame_join(tf_frame_split(tf_frame))


def topic_normalize(topic):
    """
    Function for normalizing a topic address string.

    :param topic: (string) String of a single ROS topic address.

    :return: (string) A standardized topic address.
    """

    return '/' + tf_frame_normalize(topic)


def tf_frame_eq(tf1, tf2):
    """
    Function for determining whether two TF chains are equal by ignoring slashes

    :param tf1: (string) First TF frame chain
    :param tf2: (string) Second TF frame chain

    :return: (bool) True if tf1 and tf2 represent the same path ignoring slashes
    """

    tf1_list = tf_frame_normalize(tf1)
    tf2_list = tf_frame_normalize(tf2)

    eq = tf1_list == tf2_list
    return eq


def tf_frames_eq(tf1_p, tf1_c, tf2_p, tf2_c):
    """
    Function for determining whether two TF chains are equal by ignoring slashes

    :param tf1_p: (string) Parent frame of first TF chain
    :param tf1_c: (string) Child frame of first TF chain
    :param tf2_p: (string) Parent frame of second TF chain
    :param tf2_c: (string) Child frame of second TF chain

    :return: (bool) True if tf1 and tf2 represent the same transform
    """

    return tf_frame_eq(tf1_p, tf2_p) and tf_frame_eq(tf1_c, tf2_c)


def world2map(point, map_origin, delta):
    """
    Convert from world units to discrete cell coordinates.

    :param point: (np.ndarray) X and Y position in world coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).

    :return: (np.ndarray) Integer-valued coordinates in map units. I.e.: cell indexes corresponding to x and y.
    """

    int_point = point - map_origin
    int_point /= delta
    int_point = np.round(int_point)

    return int_point.astype(np.int)


def map2world(int_point, map_origin, delta, rounded=False):
    """
    Convert from discrete map cell coordinates to world units.

    :param int_point: (np.ndarray) Row and Column indices in map coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).
    :param rounded: (bool)[Default: False] Round the resulting point up to an order of magnitude smaller
                                           than the resolution if True.
                                           Useful for repeatability when computing the center coordinates of cells.

    :return: (np.ndarray) X and Y position in world coordinates.
    """

    point = delta * np.ones_like(int_point)
    point = np.multiply(point, int_point)
    point += map_origin

    if rounded:
        decimals = np.log10(delta)
        if decimals < 0:
            decimals = int(np.ceil(-decimals) + 2)
            point = np.round(point, decimals)

    return point


def cell_centerpoint(point, map_origin, delta):
    """
    Gives the center point in world coordinates of the cell corresponding to a given point in world coordinates.

    :param point: (np.ndarray) X and Y position in world coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).

    :return: (np.ndarray) X and Y position of the cell's center in world coordinates.
    """

    int_point = world2map(point, map_origin, delta)
    cnt_point = map2world(int_point, map_origin, delta, rounded=True)

    return cnt_point


def to_np(value):
    """
    Converts scalars and lists to numpy arrays if they aren't yet.

    :param value: (number|list) Value to be converted to a numpy array.

    :return: (np.ndarray) Numpy array of the input value.
    """

    if not isinstance(value, np.ndarray):
        if not isinstance(value, list):
            value = [value]
        return np.array(value)

    return value


def normalize_angles(angles):
    """
    Normalizes angle(s) to the interval (-pi, pi]

    :param angles: (np.ndarray) Angle(s) to be normalized.

    :return: (np.ndarray) Angles normalized to (-pi, pi] interval
    """

    # Get the angles modulo 2pi. Angles are now in [0, 2pi] range
    tmp_angles = angles % (2 * np.pi)
    # Make angles greater than pi negative to be in (-pi, pi] range
    tmp_angles[tmp_angles > np.pi] -= 2 * np.pi

    return tmp_angles
