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
from tf.transformations import decompose_matrix, \
    quaternion_from_euler, quaternion_multiply

# ROS Message Libraries
from std_msgs.msg import Bool as BoolMessage
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64

# Math Libraries
import numpy as np

# OS Libraries
import os
import os.path
import datetime
import time

from map_simulator.utils import tf_frames_eq, mkdir_p
from map_simulator.geometry.transform import tf_msg_to_matrix, quaternion_axis_angle


class PoseErrorCalculator:

    def __init__(self):
        """
        Initialize the PoseErrorCalculator object, the ROS node,
        get the parameters and keep the node alive (spin)
        """

        rospy.init_node('pose_error_calc')

        # Error Accumulators
        self._cum_trans_err = 0
        self._cum_rot_err = 0
        self._cum_tot_err = 0

        # Buffers for current and last step's transformation matrices
        self._curr_gt_mb_pose = None
        self._curr_sl_mo_pose = None
        self._curr_sl_ob_pose = None
        self._last_gt_mb_pose = None
        self._last_sl_ob_pose = None
        # relative rotation quaternion (starting as zero-rotation)
        self._last_rot_err_q = quaternion_from_euler(0, 0, 0)
        # and info
        self._curr_seq = None
        self._curr_ts = None
        self._curr_processed = False
        self._last_seq = None
        self._last_ts = None
        self._last_processed = False

        self._first_pose = True
        self._loc_only_recvd = False

        # Weight factor for rotational error
        self._lambda = rospy.get_param("~lambda", 0.1)

        # Publish and save boolean options
        self._publish_error = rospy.get_param("~pub_err", True)
        self._log_error = rospy.get_param("log_err", True)

        if not (self._publish_error or self._log_error):
            rospy.logerr("Neither publishing nor logging. Why call me then? Exiting.")
            rospy.signal_shutdown("Nothing to do. Shutting down. Check _pub_err and _log_err parameters.")

        # TF Frames
        self._map_frame = rospy.get_param("~map_frame", "map")
        self._odom_frame = rospy.get_param("~odom_frame", "odom")
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._gt_odom_frame = rospy.get_param("~gt_odom_frame", "GT/odom")
        self._gt_base_frame = rospy.get_param("~gt_base_frame", "GT/base_link")

        # CSV Log Path parameters
        default_path = os.path.join("~", "Desktop")
        default_path = os.path.join(default_path, "FMP_logs")
        log_dir = rospy.get_param("~log_dir", default_path)
        log_dir = os.path.expandvars(os.path.expanduser(log_dir))
        # CSV Log File parameters
        err_prefix = rospy.get_param("~err_prefix", "pose_err")
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        default_err_file = err_prefix + '_' + timestamp + '.csv'
        default_err_file = os.path.join(log_dir, default_err_file)
        default_loc_err_file = err_prefix + '_loc_' + timestamp + '.csv'
        default_loc_err_file = os.path.join(log_dir, default_loc_err_file)
        self._err_file = rospy.get_param("~err_file", default_err_file)
        self._loc_err_file = rospy.get_param("~loc_err_file", default_loc_err_file)
        self._current_err_file = self._err_file

        # CSV row and column delimiters
        self._newline = rospy.get_param("~newline", "\n")
        self._delim = rospy.get_param("~delim", ",")

        if self._log_error:
            if not os.path.exists(log_dir):
                mkdir_p(log_dir)

            self._init_log()

        # Subscribers / Publishers
        rospy.Subscriber("/tf", TFMessage, self._tf_callback)
        rospy.Subscriber("/doLocOnly", BoolMessage, self._loc_only_callback)

        rospy.on_shutdown(self._shutdown_callback)

        if self._publish_error:
            self._tra_err_pub = rospy.Publisher("tra_err", Float64, latch=True, queue_size=1)
            self._rot_err_pub = rospy.Publisher("rot_err", Float64, latch=True, queue_size=1)
            self._tot_err_pub = rospy.Publisher("tot_err", Float64, latch=True, queue_size=1)

        rospy.spin()

    def _process_last_pose(self, ignore_eq_seq=False):
        """
        Computes the relative pose between the Ground Truth and the SLAM corrected pose.
        Computes the translation and rotational errors between the two and saves and publishes them.
        If the pose has already been processed or there was no sequence change, it does nothing.

        :param ignore_eq_seq: If True, it ignores the fact that last_seq and curr_seq might be equal and processes the
                              last pose anyways.

        :return: None
        """

        if not ignore_eq_seq and self._curr_seq == self._last_seq:
            return

        if self._last_processed:
            return

        if self._curr_sl_mo_pose is None or self._last_sl_ob_pose is None or self._last_gt_mb_pose is None:
            return

        # GT map->base Transform
        gt_mb = self._last_gt_mb_pose
        # ODO map->base Transform (= SLAM odo->base Transform)
        od_mb = self._last_sl_ob_pose
        # SLAM map->base Transform (= SLAM (map->odo) x (odo->base))
        sl_mb = np.dot(self._curr_sl_mo_pose, self._last_sl_ob_pose)
        # REL base->base Transform (relative pose between SLAM and GT bases
        rl_bb = np.dot(sl_mb, np.linalg.inv(gt_mb))

        _, _, rl_bb_r, rl_bb_t, _ = decompose_matrix(rl_bb)

        # Translational Error as squared euclidean distance
        trans_error = np.matmul(rl_bb_t, rl_bb_t)
        self._cum_trans_err += trans_error

        # Convert rotation to quaternion
        rl_bb_q = quaternion_from_euler(rl_bb_r[0], rl_bb_r[1], rl_bb_r[2])

        # Rotational Error as a single angle (rotation around single arbitrary axis)
        _, rot_error = quaternion_axis_angle(rl_bb_q)

        # Accumulate rotational errors by compounding the rotations
        self._last_rot_err_q = quaternion_multiply(self._last_rot_err_q, rl_bb_q)
        _, cum_rot_err = quaternion_axis_angle(self._last_rot_err_q)
        self._cum_rot_err = cum_rot_err

        # Compute total error as the weighted sum of translational and rotational errors.
        tot_error = trans_error + self._lambda * rot_error
        # Compute the cumulative total error as the weighted sum of cumulative translational and rotational errors.
        # TODO: verify this makes mathematical sense:
        self._cum_tot_err = self._cum_trans_err + self._lambda * self._cum_rot_err

        # Append to file if configured to do so
        if self._log_error:
            self._append_row(self._last_seq, self._last_ts, gt_mb, od_mb, sl_mb, rl_bb,
                             trans_error, rot_error, tot_error)

        # Publish error messages if configured to do so
        if self._publish_error:
            tra_err_msg = Float64()
            rot_err_msg = Float64()
            tot_err_msg = Float64()

            tra_err_msg.data = trans_error
            rot_err_msg.data = rot_error
            tot_err_msg.data = tot_error

            self._tra_err_pub.publish(tra_err_msg)
            self._rot_err_pub.publish(rot_err_msg)
            self._tot_err_pub.publish(tot_err_msg)

        # Set the pose to already processed to avoid duplicate entries, in case we end up here again for some reason
        self._last_processed = True

    def _push_pose(self, seq=None, ts=None, sl_ob_pose=None, gt_mb_pose=None, processed=None):
        """
        Push the current pose to the last pose and its info.
        If all parameters are not none, then set those as the current pose. Otherwise, retain the current pose as well.

        :param seq: (int)[Default: None] Latest Pose sequence.
        :param ts: (rospy.Time)[Default: None] Timestamp of the latest Ground Truth Pose message.
        :param sl_ob_pose: (np.ndarray)[Default: None] SLAM:odom->base transform matrix.
        :param gt_mb_pose: (np.ndarray)[Default: None] GT:map->base transform matrix.
        :param processed: (bool)[Default: None] Pose has been already processed if True.

        :return: None
        """

        self._last_seq = self._curr_seq
        self._last_ts = self._curr_ts
        self._last_processed = self._curr_processed
        self._last_sl_ob_pose = self._curr_sl_ob_pose
        self._last_gt_mb_pose = self._curr_gt_mb_pose

        if seq is not None and ts is not None and processed is not None and \
                sl_ob_pose is not None and gt_mb_pose is not None:
            self._curr_seq = seq
            self._curr_ts = ts
            self._curr_processed = processed
            self._curr_sl_ob_pose = sl_ob_pose
            self._curr_gt_mb_pose = gt_mb_pose

    def _tf_callback(self, msg):
        """
        Function called whenever a TF message is received.
        It tries to store the transforms for the Ground Truth pose (map->GT/base_link),
        and the SLAM pose (map->odom, odom->base_link), and only process them once the
        ground truth message's sequence changes to use the latest poses and allow the
        the SLAM algorithm to register all scans and compute the correction transform.
        If configured, it will publish the error as a float64 message and log to a CSV file.

        :param msg: (tf2_msgs.TFMessage) TF Messages

        :return: None
        """

        seq_chgd = False

        gt_mo_tf = None
        gt_ob_tf = None

        gt_mb_tf = None
        sl_ob_tf = None

        seq = -1
        ts = None

        # Collect transform information
        for transform in msg.transforms:
            seq = transform.header.seq
            ts = transform.header.stamp
            p_frame = transform.header.frame_id
            c_frame = transform.child_frame_id

            # If transform is SLAM:map->odom
            if tf_frames_eq(p_frame, c_frame, self._map_frame, self._odom_frame):
                seq_chgd = False
                self._curr_sl_mo_pose = tf_msg_to_matrix(transform)

            # If transform is GT:map->odom
            elif tf_frames_eq(p_frame, c_frame, self._map_frame, self._gt_odom_frame):
                gt_mo_tf = tf_msg_to_matrix(transform)

            # If transform is GT:odom->base
            elif tf_frames_eq(p_frame, c_frame, self._gt_odom_frame, self._gt_base_frame):
                gt_ob_tf = tf_msg_to_matrix(transform)

            # If transform is SLAM:odom->base
            elif tf_frames_eq(p_frame, c_frame, self._odom_frame, self._base_frame):
                sl_ob_tf = tf_msg_to_matrix(transform)

            # If we got all we need, compute the GT:map->base transform and ignore the other transforms in the message
            if gt_mo_tf is not None and gt_ob_tf is not None and sl_ob_tf is not None:
                seq_chgd = self._curr_seq != seq
                gt_mb_tf = np.dot(gt_mo_tf, gt_ob_tf)
                break

        # Compute error only if a new Ground Truth pose was published (seq_chgd)
        # and we actually acquired the necessary transforms.
        if seq_chgd and gt_mb_tf is not None and sl_ob_tf is not None and self._curr_sl_mo_pose is not None:
            self._push_pose(seq, ts, sl_ob_tf, gt_mb_tf, False)
            # Don't process first pose, until a new one comes in.
            if self._first_pose:
                self._first_pose = False
            else:
                self._process_last_pose()

            # If we received a doLocOnly message, process the current pose, reset accumulators and start a new file.
            if self._loc_only_recvd:
                self._proc_and_reset_for_loc()

    def _proc_and_reset_for_loc(self):
        """
        Function called for processing the last mapping pose, resetting the error accumulators
        and switching the logging to a new file.

        :return: None
        """
        if not self._loc_only_recvd:
            return

        self._push_pose()
        self._process_last_pose(ignore_eq_seq=True)
        self._curr_processed = True

        # Reset cumulative errors
        self._cum_trans_err = 0
        self._cum_rot_err = 0
        self._cum_trans_err = 0
        self._last_rot_err_q = quaternion_from_euler(0, 0, 0)

        self._current_err_file = self._loc_err_file

        self._init_log()

        self._loc_only_recvd = False

    def _loc_only_callback(self, msg):
        """
        Sets up a flag so that the next time a GT pose is received, _proc_and_reset_for_loc is called.

        :param msg: (std_messages.Bool) doLocOnly Message.

        :return: None
        """
        self._loc_only_recvd = msg.data

    def _shutdown_callback(self):
        """
        Callback for when the node is starting to shut down.
        It processes the poses still remaining in buffer before shutting down.

        :return: None
        """

        rospy.loginfo("Saving last poses and shutting down.")
        self._process_last_pose()  # Save info to file
        self._push_pose()
        self._process_last_pose(ignore_eq_seq=True)  # Save last pose to file

    def _append_row(self, seq, ts, gt_tf, odo_tf, slam_tf, rel_tf, trans_err, rot_err, tot_err):
        """
        Append a row to the CSV file with the poses and errors

        :param seq: (int) Sequence number of the Ground Truth Pose message.
        :param ts: (rospy.Time) Timestamp of the Ground Truth Pose message.
        :param gt_tf: (np.ndarray) GT:map->base transform matrix.
        :param odo_tf: (np.ndarray) ODOM:map->base (= odom->base) transform matrix.
        :param slam_tf: (np.ndarray) SLAM:map->base transform matrix.
        :param rel_tf: (np.ndarray) Relative transform between GT:base->SLAM:Base transform matrix.
        :param trans_err: (float) Step translational error.
        :param rot_err: (float) Step rotational error.
        :param tot_err: (float) Step total error.

        :return: None
        """

        if not self._log_error:
            return

        _,  _,  gt_rot, gt_tra, _ = decompose_matrix(gt_tf)
        gt_p_x, gt_p_y, gt_p_z = gt_tra
        gt_r_x, gt_r_y, gt_r_z = gt_rot
        _,  _,  od_rot, od_tra, _ = decompose_matrix(odo_tf)
        od_p_x, od_p_y, od_p_z = od_tra
        od_r_x, od_r_y, od_r_z = od_rot
        _,  _,  sl_rot, sl_tra, _ = decompose_matrix(slam_tf)
        sl_p_x, sl_p_y, sl_p_z = sl_tra
        sl_r_x, sl_r_y, sl_r_z = sl_rot
        _,  _,  rl_rot, rl_tra, _ = decompose_matrix(rel_tf)
        rl_p_x, rl_p_y, rl_p_z = rl_tra
        rl_r_x, rl_r_y, rl_r_z = rl_rot

        ts_time = time.localtime(ts.secs)
        ts_yy = ts_time.tm_year
        ts_mm = ts_time.tm_mon
        ts_dd = ts_time.tm_mday
        ts_h = ts_time.tm_hour
        ts_m = ts_time.tm_min
        ts_s = ts_time.tm_sec
        ts_ns = ts.nsecs

        row = [
            seq,
            ts_yy, ts_mm, ts_dd, ts_h, ts_m, ts_s, ts_ns,
            gt_p_x, gt_p_y, gt_p_z, gt_r_x, gt_r_y, gt_r_z,
            od_p_x, od_p_y, od_p_z, od_r_x, od_r_y, od_r_z,
            sl_p_x, sl_p_y, sl_p_z, sl_r_x, sl_r_y, sl_r_z,
            rl_p_x, rl_p_y, rl_p_z, rl_r_x, rl_r_y, rl_r_z,
            trans_err, self._cum_trans_err,
            rot_err, self._cum_rot_err,
            tot_err, self._cum_tot_err
        ]

        row_str = self._delim.join([str(x) for x in row])
        row_str += self._newline

        rospy.loginfo("Adding pose with seq. {} to file.".format(seq))

        with open(self._current_err_file, 'a') as f:
            f.write(row_str)

    def _init_log(self):
        """
        Initialize the CSV log file by adding the column headers.

        :return: None
        """

        if not self._log_error:
            return

        rospy.loginfo("Saving error log to {}".format(self._err_file))

        col_head1 = [
            "SEQ",
            "TIME STAMP", "", "", "", "", "", "",
            "GT POSE", "", "", "", "", "",
            "PURE ODOMETRY POSE", "", "", "", "", "",
            "SLAM POSE", "", "", "", "", "",
            "RELATIVE POSE /GT/base_link->/SLAM/base_link", "", "", "", "", "", "",
            "ERROR", "", "", "", "", ""
        ]

        col_head2 = [
            "",
            "", "", "", "", "", "", "",
            "Pose", "", "", "Orientation", "", "",
            "Pose", "", "", "Orientation", "", "",
            "Pose", "", "", "Orientation", "", "",
            "Pose", "", "", "Orientation", "", "",
            "Translational", "", "Angular", "",
            "Total (err_t + lambda * err_rot)[lambda = " + str(self._lambda) + "[m^2/rad^2]]", ""
        ]

        col_head3 = [
            "",
            "Date", "", "", "Time", "", "", "",
            "x", "y", "z", "Roll (x)", "Pitch (y)", "Yaw (z)",
            "x", "y", "z", "Roll (x)", "Pitch (y)", "Yaw (z)",
            "x", "y", "z", "Roll (x)", "Pitch (y)", "Yaw (z)",
            "x", "y", "z", "Roll (x)", "Pitch (y)", "Yaw (z)",
            "Step", "Cumulative",
            "Step", "Cumulative",
            "Step", "Cumulative"
        ]

        col_head4 = [
            "",
            "[Y]", "[M]", "[D]", "[h]", "[m]", "[s]", "[ns]",
            "[m]", "[m]", "[m]", "[rad]", "[rad]", "[rad]",
            "[m]", "[m]", "[m]", "[rad]", "[rad]", "[rad]",
            "[m]", "[m]", "[m]", "[rad]", "[rad]", "[rad]",
            "[m]", "[m]", "[m]", "[rad]", "[rad]", "[rad]",
            "[m]", "[m]",
            "[rad]", "[rad]",
            "[m^2]", "[m^2]",
        ]

        col_headers = [col_head1, col_head2, col_head3, col_head4]

        csv_header = self._newline.join([self._delim.join(col_header) for col_header in col_headers])
        csv_header += self._newline

        with open(self._current_err_file, 'w') as f:
            f.write(csv_header)
