#! /usr/bin/env python
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

import os
from os import path

import sys
import subprocess
from datetime import datetime

from map_simulator.utils import mkdir_p


def get_arg(dic, key, default, add_if_not_in=True, valid_vals=None):
    """
    Function for checking if an argument is set in a dictionary of ROS-style arguments (key:=value).
    If it is defined, return the defined value (if within the defined valid_vals).
    If not defined, return the default value. If add_if_not_in is True,
    then the value will also be added to the dictionary.

    :param dic: (dict) Dictionary containing the arg_name, arg_value pair
    :param key: (string) Argument name to search in the argument dictionary.
    :param default: () Default value to take if key is not defined in the dictionary, or if the value is not within
                       the possible valid_vals.
    :param add_if_not_in: (bool)[Default: True] Add the default value to the dictionary if it was not defined before,
                                                or if the defined value in the dictionary does not match any of
                                                the valid_vals (if not None).
    :param valid_vals: (list)[Default: None] For discrete or enumeration-type arguments, defines the list of all
                                             possible values. Used to check if the value stored in the dictionary is
                                             valid. If None, then no validation check is performed.

    :return: () Either the value stored in the dictionary under key, or the default value.
    """

    not_in = True

    if key not in dic:
        val = default
    else:
        val = dic[key]
        if valid_vals is not None:
            if val not in valid_vals:
                val = default
            else:
                not_in = False
        else:
            not_in = False

    if not_in and add_if_not_in:
        dic[key] = val

    return val


if __name__ == "__main__":
    """
    Wrapper script for the experiment.launch launch file.

    It helps for redirecting the log files to the experiment save directory,
    and in setting the time stamp for the directory's name,
    as these two seemingly simple tasks proved impossible to do directly using ROS's launch infrastructure.
    """
    args = sys.argv[1:]
    args_dict = dict(arg.split(":=") for arg in args)

    path_save = get_arg(args_dict, 'path_save', None, add_if_not_in=False)
    if path_save is None:
        exp_pfx = get_arg(args_dict, 'exp_pfx', 'exp')
        ts = get_arg(args_dict, 'ts', datetime.now().strftime('%y%m%d_%H%M%S'))
        mm = get_arg(args_dict, 'mm', 'rfl', valid_vals=['rfl', 'dec'])
        pw = get_arg(args_dict, 'pw', 'cmh', valid_vals=['cmh', 'ml', 'fsm'])

        path_save = path.join(path.expanduser("~"), "Desktop", "Experiments", "IndividualExperiments",
                              "{}_{}_{}_{}".format(exp_pfx, ts, mm, pw))

    # Create path for log files
    log_path = path.join(path_save, "log")
    if not path.exists(log_path):
        mkdir_p(log_path)

    # Create a symlink to the latest experiment
    link_dest = path.join(path_save, "..", "latest")
    if path.islink(link_dest):
        os.remove(link_dest)

    os.symlink(path_save, link_dest)

    # Setup environment with path for logging
    env = os.environ.copy()
    env['ROS_LOG_DIR'] = log_path

    cmd = "roslaunch map_simulator experiment_real_data.launch"
    if args_dict:
        cmd += " " + " ".join(["{}:=\"{}\"".format(k, v) for k, v in args_dict.items()])

    subprocess.call(cmd, shell=True, env=env)
