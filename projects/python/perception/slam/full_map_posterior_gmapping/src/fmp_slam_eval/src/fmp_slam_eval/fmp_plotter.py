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

# OS Libraries
import os
import os.path
import datetime

# Data Structure Libraries
from collections import deque

# ROS Libraries
import rospy

# ROS Messages
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from gmapping.msg import doubleMap, mapModel

# Math Libraries
import numpy as np
import numpy.ma as ma
from cv_bridge import CvBridge

import matplotlib
import matplotlib.pyplot as plt

# Project Libraries
from fmp_slam_eval.map_colorizer import MapColorizer
from fmp_slam_eval.enums import DiscreteStates as DiSt
from map_simulator.utils import map_msg_to_numpy, map_msg_extent, mkdir_p

# Use non-interactive plotting back-end due to issues with rospy.spin()
matplotlib.use('SVG')


class FMPPlotter:
    """
    Class for plotting/coloring different statistics from the Full Map Posterior distribution
    and publishing them as images or saving them in files.
    """

    def __init__(self):
        """
        Constructor
        """

        rospy.init_node('fmp_plot')

        # Object for pseudo-coloring and plotting the maps
        self._map_colorizer = MapColorizer()

        self._sub_topic_map_model = "map_model"
        self._sub_topic_fmp_alpha = "fmp_alpha"
        self._sub_topic_fmp_beta = "fmp_beta"

        self._map_model = None

        # TODO: this two guys:
        # do_img_raw  = rospy.get_param("~img_raw" , False)
        # do_img_fmp  = rospy.get_param("~img_fmp" , False)
        do_img_stat = rospy.get_param("~img_stat", False)
        do_img_mlm = rospy.get_param("~img_mlm", False)
        do_img_para = rospy.get_param("~img_para", False)

        self._pub_img = rospy.get_param("~pub_img", False)
        self._topic_prefix = rospy.get_param("~pub_topic_prefix", "/fmp_img/")

        self._save_img = rospy.get_param("~save_img", False)
        self._resolution = rospy.get_param("~resolution", 300)

        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        path_prefix = rospy.get_param("~path_prefix", "exp")
        default_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        default_path = os.path.join(default_path, 'FMP_img')
        default_path = os.path.join(default_path, path_prefix + "_" + timestamp)
        save_dir = rospy.get_param("~save_dir", default_path)
        save_dir = os.path.expanduser(save_dir)
        save_dir = os.path.expandvars(save_dir)
        save_dir = os.path.normpath(save_dir)
        self._save_dir = save_dir

        # Image config dictionary
        sub_img_stat_mean_cfg = {"key": "mean", "dir": os.path.join("stats", "mean"), "file_prefix": "mean",
                                 "topic": "stats/mean", "calc_f": self._calc_mean}
        sub_img_stat_var_cfg = {"key": "var", "dir": os.path.join("stats", "var"), "file_prefix": "var",
                                "topic": "stats/var", "calc_f": self._calc_var}
        img_stat_cfg = {"do": do_img_stat, "img": [sub_img_stat_mean_cfg, sub_img_stat_var_cfg]}

        sub_img_mlm_cfg = {"key": "mlm", "dir": "mlm", "file_prefix": "mlm",
                           "topic": "mlm", "calc_f": self._calc_mlm}

        img_mlm_cfg = {"do": do_img_mlm, "img": [sub_img_mlm_cfg]}

        sub_img_par_alpha_cfg = {"key": "alpha", "dir": os.path.join("param", "alpha"), "file_prefix": "alpha",
                                 "topic": "param/alpha", "calc_f": self._calc_para_alpha}
        sub_img_par_beta_cfg = {"key": "beta", "dir": os.path.join("param", "beta"), "file_prefix": "beta",
                                "topic": "param/beta", "calc_f": self._calc_para_beta}
        img_par_cfg = {"do": do_img_para, "img": [sub_img_par_alpha_cfg, sub_img_par_beta_cfg]}

        self._img_cfg = {
            "stat": img_stat_cfg,
            "mlm": img_mlm_cfg,
            "par": img_par_cfg
        }

        fmp_param_sub_required = False

        # Queues for storing messages
        self._alpha_beta_dict = {}
        self._alpha_beta_queue = deque()

        # Max and Min dictionaries for stabilizing the color scales for continuous values
        self._max_values = {}
        self._min_values = {}

        # Create Publishers
        self._publishers = {}

        for img_set_key, img_set_cfg in self._img_cfg.items():
            fmp_param_sub_required = fmp_param_sub_required or img_set_cfg['do']
            if self._pub_img and img_set_cfg['do']:
                for img_cfg in img_set_cfg['img']:
                    key = img_cfg['key']
                    topic = self._topic_prefix + img_cfg['topic']
                    self._publishers[key] = rospy.Publisher(topic, Image, latch=True, queue_size=1)

        something_to_do = (self._pub_img or self._save_img) and fmp_param_sub_required
        # Don't start the node if not needed...
        if not something_to_do:
            rospy.logerr("Nothing to do here! Why though?!?")
            rospy.logdebug("Setting values:")
            rospy.logdebug("\tpub_img: {}, save_img: {}".format(self._pub_img, self._save_img))
            rospy.logdebug("\tdo_img_stat: {}, do_img_mlm: {}, do_img_para: {}".format(do_img_stat,
                                                                                       do_img_mlm, do_img_para))
            rospy.logdebug("\tsomething_to_do: {}".format(something_to_do))
            rospy.signal_shutdown('Nothing to do')
            return

        # Create Subscribers
        # To map model
        rospy.Subscriber(self._sub_topic_map_model, mapModel, self._map_model_callback)
        # To alpha and beta parameters (if publishing or saving images, and at least one image is generated)
        if (self._pub_img or self._save_img) and fmp_param_sub_required:
            rospy.Subscriber(self._sub_topic_fmp_alpha, doubleMap, self._map2d_alpha_callback, queue_size=1)
            rospy.Subscriber(self._sub_topic_fmp_beta, doubleMap, self._map2d_beta_callback, queue_size=1)

        # Create save path if not exists
        if self._save_img and fmp_param_sub_required:
            if not os.path.exists(self._save_dir):
                mkdir_p(self._save_dir)

        self._busy = False  # Thread lock flag for plot_from_queue
        rospy.Timer(rospy.Duration(1), self._plot_from_queue)

        rospy.spin()

    def _plot_from_queue(self, event):
        """
        Function called periodically to check if there are any maps in the queue to be plotted.
        While there are still alpha and beta maps stored in the queue, it will plot the configured images.

        :param event: Caller event. Unused except for logging.

        :return: None
        """

        if self._busy:
            rospy.loginfo("Another thread is already plotting. Caller: {}".format(event))

        else:
            self._busy = True

            while self._alpha_beta_queue:
                seq = self._alpha_beta_queue.popleft()
                self._plot(seq, self._alpha_beta_dict[seq])
                del self._alpha_beta_dict[seq]

        self._busy = False

    def _plot(self, seq, dic):
        """
        Generates the desired images and plots for a given sequence of alpha and beta maps.

        :param seq: (int) Sequence number of the received maps
        :params dic: (dict) Dictionary containing the alpha and beta maps, as well as their prior values.
                            It should be formatted as:
                            dic = {'alpha': {'prior': (int), 'map': (2D np.ndarray)},
                                   'beta' : {'prior': (int), 'map': (2D np.ndarray)}}

        :return: None
        """

        if not self._pub_img and not self._save_img:
            return

        extent_a = dic['alpha']['extent']
        extent_b = dic['beta']['extent']
        if extent_a != extent_b:
            raise ValueError("Map extent of alpha {} differs from beta {}!".format(extent_a, extent_b))

        self._map_colorizer.set_wm_extent(extent_a)

        alpha = dic['alpha']['map'] + dic['alpha']['prior']
        beta = dic['beta']['map'] + dic['beta']['prior']

        for img_set_key, img_set_cfg in self._img_cfg.items():
            if img_set_cfg['do']:
                rospy.loginfo('Plotting %s', img_set_key)
                for img_cfg in img_set_cfg['img']:

                    img_key = img_cfg['key']
                    img_calc = img_cfg['calc_f']

                    rospy.loginfo("\tComputing continuous and discrete images for %s.", img_key)

                    # Compute the images to plot using the configured calculation_function ('calc_f')
                    img_cont, img_disc, ds_list, v_min, v_max, occ, log_scale = img_calc(alpha, beta)

                    self._map_colorizer.set_disc_state_list(ds_list)
                    self._map_colorizer.set_cont_bounds(img_cont, v_min=v_min, v_max=v_max, occupancy_map=occ,
                                                        log_scale=log_scale)

                    rgba_img = self._map_colorizer.colorize(img_cont, img_disc)

                    del img_cont
                    del img_disc

                    if self._save_img:
                        path = os.path.join(self._save_dir, img_cfg['dir'])

                        if not os.path.exists(path):
                            mkdir_p(path)

                        filename = img_cfg['file_prefix'] + '_s' + str(seq)
                        raw_filename = 'raw_' + filename + '.png'
                        filename = filename + '.svg'
                        mlp_path = os.path.join(path, filename)
                        raw_path = os.path.join(path, raw_filename)

                        fig, ax = plt.subplots(figsize=[20, 20])
                        ax.imshow(rgba_img, extent=extent_a)
                        self._map_colorizer.draw_cb_cont(fig)
                        if ds_list:
                            self._map_colorizer.draw_cb_disc(fig)

                        rospy.loginfo("\t\tSaving image %s to %s.", img_key, mlp_path)
                        plt.savefig(mlp_path, bbox_inches='tight', dpi=self._resolution)
                        plt.close()
                        del fig
                        del ax

                        rospy.loginfo("\t\tSaving image %s to %s.", img_key, raw_path)
                        plt.imsave(raw_path, rgba_img, vmin=0, vmax=1)
                        plt.close()

                        rospy.loginfo("\t\tImages saved.")

                    if self._pub_img:
                        publisher = self._publishers[img_key]

                        rospy.loginfo("\t\tGenerating image message to %s.", img_key)

                        rgba_img = 255 * rgba_img
                        rgba_img = rgba_img.astype(np.uint8)

                        image_msg_head = Header()

                        image_msg_head.seq = seq
                        image_msg_head.stamp = rospy.Time.now()
                        image_msg_head.frame_id = 'map'

                        br = CvBridge()
                        image_msg = br.cv2_to_imgmsg(rgba_img, encoding="rgba8")
                        del rgba_img
                        image_msg.header = image_msg_head

                        publisher.publish(image_msg)
                        del image_msg

                        rospy.loginfo("\t\tImage published.")

    def _map_model_callback(self, msg):
        """
        Method called when receiving a map model type. It just sets the local field with the message's value.

        :param msg: (gmapping.mapModel) An integer stating the type of map model used by the SLAM algorithm and some
                                        constants for comparisons.

        :return: None
        """

        mm = msg.map_model
        mm_str = ''

        if mm == mapModel.REFLECTION_MODEL:
            mm_str = 'Reflection Model'
        elif mm == mapModel.DECAY_MODEL:
            mm_str = 'Exponential Decay Model'
        else:
            rospy.logerr('No idea what kind of model %d is! Going with Reflection Model.', mm)
            mm = mapModel.REFLECTION_MODEL

        rospy.loginfo("Received Map Model: (%d, %s)", mm, mm_str)

        self._map_model = mm

    def _add_to_dict(self, a_b, msg):
        """
        Adds the received map and prior to the object's buffer dictionary.

        :param a_b: (string) Indicates which of the parameters has been received: "alpha"|"beta"
        :param msg: (gmapping.doubleMap) Double Map message containing the prior and map parameters.

        :return: None
        """

        seq = msg.header.seq

        map_dict = {
            a_b: {
                'map': map_msg_to_numpy(msg),
                'extent': map_msg_extent(msg),
                'prior': msg.param
            }
        }

        if a_b == 'alpha':
            b_a = 'beta'
        else:
            b_a = 'alpha'

        rospy.loginfo('Received msg for {} with seq {}'.format(a_b, seq))
        if seq in self._alpha_beta_dict:
            self._alpha_beta_dict[seq][a_b] = map_dict[a_b]
            if b_a in self._alpha_beta_dict[seq]:
                rospy.loginfo('Collected alpha/beta info for seq {}'.format(seq))
                self._alpha_beta_queue.append(seq)
        else:
            self._alpha_beta_dict[seq] = map_dict

    def _map2d_alpha_callback(self, msg):
        """
        Method called when receiving a map with the alpha parameters of the full posterior map distribution.
        It adds the received map to the buffer dictionary until both parameter maps have been received.

        :param msg: (gmapping.doubleMap) A floating point gmapping map message.

        :return: None
        """

        self._add_to_dict('alpha', msg)

    def _map2d_beta_callback(self, msg):
        """
        Method called when receiving a map with the beta parameters of the full posterior map distribution.
        It adds the received map to the buffer dictionary until both parameter maps have been received.

        :param msg: (gmapping.doubleMap) A floating point gmapping map message.

        :return: None
        """

        self._add_to_dict('beta', msg)

    def _calc_mean(self, alpha, beta):
        """
        Takes the alpha and beta parameter maps and computes the mean depending on the mapping model used.

        :param alpha: (nd.array) A 2D array containing the alpha parameters of the PDF of the map posterior.
        :param beta: (nd.array) A 2D array containing the beta parameters of the PDF of the map posterior.

        :return: (tuple) A tuple consisting of:
                             * means (ma.array),
                             * special-case discrete-valued means (ma.array),
                             * list of special discrete states (list)
                             * minimum continuous value (float) for color map scaling
                             * maximum continuous value (float) for color map scaling
                             * whether the map represents occupancy (bool)
                             * whether the color scale should be logarithmic (bool)
        """

        shape = alpha.shape

        v_min = 0
        occ = True

        if self._map_model == mapModel.DECAY_MODEL:
            numerator = alpha
            denominator = beta

            undef_mask = (denominator == 0)
            zero_mask = (numerator == 0)

            all_mask = np.logical_or(undef_mask, zero_mask)

            numerator = ma.masked_array(numerator)
            numerator[all_mask] = ma.masked

            means = ma.divide(numerator, denominator)

            means_ds = ma.zeros(shape, dtype=np.int8)
            means_ds[undef_mask] = DiSt.UNDEFINED.value
            means_ds[zero_mask] = DiSt.ZERO.value
            means_ds[~all_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED, DiSt.ZERO]
            v_max = None
            log_scale = True

        elif self._map_model == mapModel.REFLECTION_MODEL:
            denominator = alpha + beta

            undef_mask = (denominator == 0)

            numerator = ma.masked_array(alpha)
            numerator[undef_mask] = ma.masked

            means = ma.divide(numerator, denominator)

            means_ds = ma.zeros(shape, dtype=np.int8)
            means_ds[undef_mask] = DiSt.UNDEFINED.value
            means_ds[~undef_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED]
            v_max = 1
            log_scale = False

        else:
            means = ma.ones(shape)
            means_ds = None
            ds_list = []
            v_max = None
            log_scale = False
            rospy.logerr('No valid map model defined!')

        return means, means_ds, ds_list, v_min, v_max, occ, log_scale

    def _calc_var(self, alpha, beta):
        """
        Takes the alpha and beta parameter maps and computes the variance depending on the mapping model used.

        :param alpha: (nd.array) A 2D array containing the alpha parameters of the PDF of the map posterior.
        :param beta: (nd.array) A 2D array containing the beta parameters of the PDF of the map posterior.

        :return: (tuple) A tuple consisting of:
                             * variances (ma.array),
                             * special-case discrete-valued variances (ma.array),
                             * list of special discrete states (list)
                             * minimum continuous value (float) for color map scaling
                             * maximum continuous value (float) for color map scaling
                             * whether the map represents occupancy (bool)
                             * whether the color scale should be logarithmic (bool)
        """

        shape = alpha.shape

        v_min = 0
        occ = False

        if self._map_model == mapModel.DECAY_MODEL:
            numerator = alpha
            denominator = np.multiply(beta, beta)

            undef_mask = (denominator == 0)
            zero_mask = (numerator == 0)

            all_mask = np.logical_or(undef_mask, zero_mask)

            numerator = ma.masked_array(numerator)
            numerator[all_mask] = ma.masked

            variances = ma.divide(numerator, denominator)

            vars_ds = ma.zeros(shape, dtype=np.int8)
            vars_ds[undef_mask] = DiSt.UNDEFINED.value
            vars_ds[zero_mask] = DiSt.ZERO.value
            vars_ds[~all_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED, DiSt.ZERO]
            v_max = None
            log_scale = True

        elif self._map_model == mapModel.REFLECTION_MODEL:
            a_plus_b = alpha + beta
            numerator = np.multiply(alpha, beta)
            denominator = np.multiply(np.multiply(a_plus_b, a_plus_b), (a_plus_b + 1))

            undef_mask = (denominator == 0)

            numerator = ma.masked_array(numerator)
            numerator[undef_mask] = ma.masked

            variances = ma.divide(numerator, denominator)

            vars_ds = ma.zeros(shape, dtype=np.int8)
            vars_ds[undef_mask] = DiSt.UNDEFINED.value
            vars_ds[~undef_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED]
            v_max = None
            log_scale = False

        else:
            variances = ma.ones(shape)
            vars_ds = None
            ds_list = []
            v_max = 1
            log_scale = False
            rospy.logerr('No valid map model defined!')

        return variances, vars_ds, ds_list, v_min, v_max, occ, log_scale

    def _calc_mlm(self, alpha, beta):
        """
        Takes the alpha and beta parameter maps and computes the most-likely map depending on the mapping model used.

        :param alpha: (nd.array) A 2D array containing the alpha parameters of the PDF of the map posterior.
        :param beta: (nd.array) A 2D array containing the beta parameters of the PDF of the map posterior.

        :return: (tuple) A tuple consisting of:
                             * most-likely map values (ma.array),
                             * special-case discrete-valued most-likely map values (ma.array),
                             * list of special discrete states (list)
                             * minimum continuous value (float) for color map scaling
                             * maximum continuous value (float) for color map scaling
                             * whether the map represents occupancy (bool)
                             * whether the color scale should be logarithmic (bool)
        """
        shape = alpha.shape

        numerator = ma.masked_array(alpha - 1)

        v_min = 0

        if self._map_model == mapModel.REFLECTION_MODEL:
            denominator = alpha + beta - 2

            undef_mask = (denominator == 0)
            n_undef_mask = ~undef_mask

            unif_mask = np.logical_and(alpha == 1, beta == 1)
            unif_mask = np.logical_and(unif_mask, n_undef_mask)
            bimod_mask = np.logical_and(alpha < 1, beta < 1)
            bimod_mask = np.logical_and(bimod_mask, n_undef_mask)

            mask = np.logical_or(undef_mask, unif_mask)
            mask = np.logical_or(mask, bimod_mask)

            numerator[mask] = ma.masked

            mlm = ma.divide(numerator, denominator)

            mlm_ds = ma.zeros(shape, dtype=np.int8)
            mlm_ds[~mask] = ma.masked
            mlm_ds[unif_mask] = DiSt.UNIFORM.value
            mlm_ds[undef_mask] = DiSt.UNDEFINED.value
            mlm_ds[bimod_mask] = DiSt.BIMODAL.value

            ds_list = [DiSt.UNDEFINED, DiSt.UNIFORM, DiSt.BIMODAL]
            v_max = 1
            log_scale = False

        elif self._map_model == mapModel.DECAY_MODEL:
            denominator = beta

            undef_mask = np.logical_or(denominator == 0, alpha < 1)
            n_undef_mask = ~undef_mask
            zero_mask = np.logical_and(numerator == 0, n_undef_mask)

            all_mask = np.logical_or(undef_mask, zero_mask)

            numerator[all_mask] = ma.masked
            mlm = ma.divide(numerator, denominator)

            mlm_ds = ma.zeros(shape, dtype=np.int8)
            mlm_ds[undef_mask] = DiSt.UNDEFINED.value
            mlm_ds[zero_mask] = DiSt.ZERO.value
            mlm_ds[~all_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED, DiSt.ZERO]
            v_max = None
            log_scale = True

        else:
            rospy.logerr('No valid map model defined!')
            mlm = ma.zeros(shape)
            mlm_ds = None
            ds_list = []
            v_max = 1
            log_scale = False

        occ = True

        return mlm, mlm_ds, ds_list, v_min, v_max, occ, log_scale

    @staticmethod
    def _calc_para_alpha(alpha, _):
        """
        Simply returns the alpha parameter map.

        :param alpha: (nd.array) A 2D array containing the alpha parameters of the PDF of the map posterior.
        :param _: Unused

        :return: (tuple) A tuple consisting of:
                             * alpha (np.ndarray),
                             * no special cases (None)
                             * no special cases (empty list)
                             * minimum continuous value (0) for color map scaling
                             * maximum continuous value (None: i.e. unbounded) for color map scaling
                             * map does not represent occupancy (False)
                             * color scale should be linear (False)
        """

        return alpha, None, [], 0, None, False, False

    @staticmethod
    def _calc_para_beta(_, beta):
        """
        Simply returns the beta parameter map.

        :param _: Unused
        :param beta: (nd.array) A 2D array containing the beta parameters of the PDF of the map posterior.

        :return: (tuple) A tuple consisting of:
                             * beta (np.ndarray),
                             * no special cases (None)
                             * no special cases (empty list)
                             * minimum continuous value (0) for color map scaling
                             * maximum continuous value (None: i.e. unbounded) for color map scaling
                             * map does not represent occupancy (False)
                             * color scale should be linear (False)
        """

        return beta, None, [], 0, None, False, False
