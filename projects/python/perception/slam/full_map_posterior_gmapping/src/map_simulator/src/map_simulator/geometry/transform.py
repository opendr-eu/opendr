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

from tf.transformations import quaternion_matrix
import numpy as np


def rotate2d(theta):
    """
    Compute a 2D rotation matrix given an angle.

    :param theta: (float) Angle to rotate about the z axis.

    :return: (ndarray) A 2x2 rotation matrix.
    """

    s, c = np.sin(theta), np.cos(theta)

    return np.array([[c, -s], [s, c]]).reshape((2, 2))


def quaternion_axis_angle(q):
    """
    Convert a rotation expressed as a quaternion into a 3D vector
    representing the axis of rotation and the rotation angle in radians.

    :param q: (list|tuple) A quaternion (x, y, z, w)

    :return: (numpy.array, float) A tuple containting a 3D numpy vector and the angle as float
    """

    w, v = q[3], q[0:2]
    theta = np.arccos(w) * 2
    return v, theta


def tf_msg_to_matrix(msg):
    """
    Converts a transform message to a transformation matrix in homogeneous coordinates.

    :param msg: (TFMessage) TF Message to be converted to a matrix.

    :return: (np.ndarray) 4x4 Transformation Matrix in Homogeneous Coordinates.
    """

    translation = msg.transform.translation
    rotation = msg.transform.rotation
    p = np.array([translation.x, translation.y, translation.z])
    q = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    transform_matrix = quaternion_matrix(q)
    transform_matrix[0:3, -1] = p
    return transform_matrix
