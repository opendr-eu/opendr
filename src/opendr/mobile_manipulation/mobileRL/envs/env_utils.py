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

import numpy as np
import rospy
from geometry_msgs.msg import Quaternion, Pose, Point
from scipy.spatial.transform import Rotation
from typing import Union, List
from visualization_msgs.msg import Marker


def quaternion_to_yaw(q: Union[List, Quaternion]) -> float:
    if isinstance(q, Quaternion):
        q = [q.x, q.y, q.z, q.w]
    yaw = Rotation.from_quat(q)
    yaw = yaw.as_euler('xyz')[..., -1]
    if not yaw.shape:
        # squeeze to a scalar if possible
        return yaw.item()
    else:
        return yaw


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(*Rotation.from_euler('xyz', [0, 0, yaw]).as_quat().tolist())


def publish_marker(namespace: str, marker_pose: Pose, marker_scale, marker_id: int, frame_id: str, geometry: str,
                   color: str = "orange", alpha: float = 1):
    assert len(marker_scale) == 3

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.get_rostime()
    marker.ns = namespace
    marker.id = marker_id
    marker.action = 0
    if geometry == "arrow":
        marker.type = Marker.ARROW
    elif geometry == "cube":
        marker.type = Marker.CUBE
    else:
        raise NotImplementedError()

    marker.pose = marker_pose
    marker.scale.x = marker_scale[0]
    marker.scale.y = marker_scale[1]
    marker.scale.z = marker_scale[2]

    assert color == "orange", "atm only orange"
    marker.color.r = 1.0
    marker.color.g = 159 / 255
    marker.color.b = 0.0
    marker.color.a = alpha

    pub = rospy.Publisher('kinematic_feasibility_py', Marker, queue_size=10)
    pub.publish(marker)


def clear_all_markers(frame_id: str):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.get_rostime()
    marker.action = 3
    pub = rospy.Publisher('kinematic_feasibility_py', Marker, queue_size=10)
    pub.publish(marker)


def calc_disc_return(rewards: list, gamma: float) -> float:
    return (gamma ** np.arange(len(rewards)) * rewards).sum()


def pose_to_list(pose: Pose) -> list:
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


def list_to_pose(l: list) -> Pose:
    assert len(l) == 7
    return Pose(Point(l[0], l[1], l[2]), Quaternion(l[3], l[4], l[5], l[6]))
