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

from os import path

from std_msgs.msg import Float64, Bool
from map_simulator.utils import mkdir_p


class ErrorDataCollector(object):

    def __init__(self):

        rospy.init_node("data_collector")

        def_file_path = path.join("~", "Desktop", "Experiments")
        file_path = rospy.get_param("~file_path", def_file_path)
        file_prefix = rospy.get_param("~file_prefix", "error_data")
        mapping_suffix = rospy.get_param("~mapping_suffix", "_map")
        localization_suffix = rospy.get_param("~localization_suffix", "_loc")

        if not path.exists(file_path):
            mkdir_p(file_path)

        filename = path.join(file_path, file_prefix)
        mapping_filename = filename + mapping_suffix
        mapping_filename = path.expanduser(mapping_filename)
        mapping_filename = path.expandvars(mapping_filename)
        localization_filename = filename + localization_suffix
        localization_filename = path.expanduser(localization_filename)
        localization_filename = path.expandvars(localization_filename)

        self._mapping_filename = mapping_filename
        self._localization_filename = localization_filename
        self._current_err_file = self._mapping_filename

        self._fs = ','
        self._rs = '\n'

        self._errors = {
            "tra": [],
            "rot": [],
            "tot": [],
        }

        rospy.Subscriber("doLocOnly", Bool, self._loc_callback)
        rospy.Subscriber("tra_err", Float64, self._err_callback, callback_args="tra")
        rospy.Subscriber("rot_err", Float64, self._err_callback, callback_args="rot")
        rospy.Subscriber("tot_err", Float64, self._err_callback, callback_args="tot")
        rospy.on_shutdown(self._shutdown_callback)

        rospy.spin()

    def _shutdown_callback(self):
        self._append_errors_to_file()

    def _loc_callback(self, msg):
        if msg.data:
            self._append_errors_to_file()
            self._current_err_file = self._localization_filename

    def _err_callback(self, msg, error_type):
        self._errors[error_type].append(msg.data)

    def _append_errors_to_file(self):
        for key, values in self._errors.items():
            file_name = self._current_err_file + "_{}.csv".format(key)
            err_row = self._fs.join([str(v) for v in values]) + self._rs

            with open(file_name, 'a') as f:
                f.write(err_row)

    @staticmethod
    def stop():
        rospy.signal_shutdown("Simulation Finished. Shutting down")

    @staticmethod
    def is_running():
        return not rospy.is_shutdown()
