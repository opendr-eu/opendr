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

import rospy

from tf.transformations import quaternion_from_euler
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point, TransformStamped

from map_simulator.utils import tf_frame_normalize, tf_frame_join

_zero_trans = Point(0, 0, 0)
_zero_rot = quaternion_from_euler(0.0, 0.0, 0)


class OdomPosePublisher:

    def __init__(self):
        rospy.init_node('odom_pose')

        self._map_frame = tf_frame_normalize(rospy.get_param('~map_frame', 'map'))
        self._odom_frame = tf_frame_normalize(rospy.get_param('~odom_frame', 'odom'))

        self._frame_prefix = rospy.get_param('~frame_prefix', 'odo')

        frame_list = rospy.get_param('~frame_list', '[base_link, laser_link]')
        frame_list = frame_list.replace('[', '')
        frame_list = frame_list.replace(']', '')
        frame_list = frame_list.strip()
        frame_list = frame_list.split(',')
        frame_list = [tf_frame_normalize(frame) for frame in frame_list]
        if self._odom_frame not in frame_list:
            frame_list.append(self._odom_frame)

        self._frame_list = frame_list

        rospy.Subscriber("tf", TFMessage, self._tf_callback, queue_size=1)

        self._pub_tf = rospy.Publisher("tf", TFMessage, queue_size=1)

        rospy.spin()

    def _tf_callback(self, msg):
        """
        Function called whenever a TF message is received.
        If the message involves a pair of desired tf frames, it will publish a map->new_odom transform,
        and republish the desired tf messages with a prefix added to the frames

        :param msg: (tf2_msgs.TFMessage) TF Messages

        :return: None
        """

        transforms = []

        seq = None
        ts = rospy.Time.now()

        for transform in msg.transforms:

            if seq is None:
                seq = transform.header.seq
            elif transform.header.seq < seq:
                seq = transform.header.seq

            if transform.header.stamp.secs < ts.secs or\
                    (transform.header.stamp.secs == ts.secs and transform.header.stamp.nsecs < ts.nsecs):
                ts = transform.header.stamp

            pframe = tf_frame_normalize(transform.header.frame_id)
            cframe = tf_frame_normalize(transform.child_frame_id)

            if pframe in self._frame_list and cframe in self._frame_list:

                new_pframe = tf_frame_normalize(tf_frame_join(self._frame_prefix, pframe))
                new_cframe = tf_frame_normalize(tf_frame_join(self._frame_prefix, cframe))

                transform.header.frame_id = new_pframe
                transform.child_frame_id = new_cframe

                transforms.append(transform)

        if transforms:
            new_tf_msg = TFMessage()

            map_odom_tf = TransformStamped()

            map_odom_tf.header.stamp = ts
            map_odom_tf.header.seq = seq
            map_odom_tf.header.frame_id = self._map_frame
            map_odom_tf.child_frame_id = tf_frame_normalize(tf_frame_join(self._frame_prefix, self._odom_frame))

            map_odom_tf.transform.translation = _zero_trans
            map_odom_tf.transform.rotation.x = _zero_rot[0]
            map_odom_tf.transform.rotation.y = _zero_rot[1]
            map_odom_tf.transform.rotation.z = _zero_rot[2]
            map_odom_tf.transform.rotation.w = _zero_rot[3]

            new_tf_msg.transforms.append(map_odom_tf)

            for transform in transforms:
                new_tf_msg.transforms.append(transform)

            self._pub_tf.publish(new_tf_msg)
