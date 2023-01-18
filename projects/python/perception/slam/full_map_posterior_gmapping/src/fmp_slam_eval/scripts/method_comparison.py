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

import rospkg

import argparse

from os import path, getpid
from time import sleep
from datetime import datetime

import itertools
from collections import OrderedDict

import multiprocessing

from map_simulator.ros_launcher import ROSLauncher


def run_exp_n_times(package, launch_file_path, iterations=1, launch_args_dict=None, log_path=None,
                    port=None, monitored_nodes=None):

    pid = getpid()
    print("\n" * 2)
    print("*" * 80)
    print("<*> STARTED Process {}, Params {}.".format(pid, launch_args_dict["path_err_coll_pfx"]))
    print("\n" * 2)

    for curr_it in range(iterations):
        print("\n" * 2)
        print("=" * 60)
        print("<*>\tProcess {}, Params {}, Iteration {}/{}.\n\n\n".format(pid, launch_args_dict["path_err_coll_pfx"],
                                                                          curr_it + 1, iterations))
        print("\n" * 2)

        # Run Launch file
        launch_args_dict["ts"] = datetime.now().strftime('%y%m%d_%H%M%S')

        launcher = ROSLauncher(package, launch_file_path, wait_for_master=False, log_path=log_path, port=port,
                               monitored_nodes=monitored_nodes)
        launcher.start(launch_args_dict)
        launcher.spin()
        sleep(1)

    print("\n" * 2)
    print("*" * 80)
    print("<*> FINISHED Process {}, Params {}.".format(pid, launch_args_dict["path_err_coll_pfx"]))
    print("\n" * 2)


def list_split(string):
    return filter(None, string.strip().split(','))


def list_parse(string, parse_type):
    return list(map(parse_type, list_split(string)))


def int_list(string):
    return list_parse(string, int)


def str_list(string):
    return list_parse(string, str)


def bool_list(string):
    return [s.lower() == 'true' for s in list_split(string)]


def float_list(string):
    return list_parse(string, float)


if __name__ == "__main__":

    launchfile_package = "fmp_slam_eval"
    pck = rospkg.RosPack()
    launch_pck_share = pck.get_path(launchfile_package)
    def_launch_file = path.join(launch_pck_share, "launch", "experiment.launch")
    slamdata_pck_share = pck.get_path('slam_datasets')

    def_file_path = path.join("~", "Desktop", "Experiments", "MethodComparison")
    def_file_path = path.expanduser(def_file_path)
    def_file_path = path.expandvars(def_file_path)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run several mapping and localization experiments and collect data")
    parser.add_argument('-i', '--iterations', action='store', type=int, default=100,
                        help='Number of times to run each experiment.')
    parser.add_argument('-m', '--moves', action='store', type=int_list,
                        default="20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,270,300",
                        help='Comma-separated list of number of movements to run the tests with.')
    parser.add_argument('-mm', '--map_models', action='store', type=str_list, default="ref,dec",
                        help='Comma-separated list of map model types.')
    parser.add_argument('-pw', '--particle_weights', action='store', type=str_list, default="cmh,ml",
                        help='Comma-separated list of particle weighting methods.')
    parser.add_argument('-pi', '--pose_improve', action='store', type=bool_list, default="False",
                        help='Comma-separated list of pose improvement options.')
    parser.add_argument('-f', '--launch_file', action='store', type=str, default=def_launch_file,
                        help='Launch file to execute.')
    parser.add_argument('-mp', '--multi_proc', action='store_true', default=True,
                        help='Run multiple processes concurrently if true.')
    parser.add_argument('-w', '--num_workers', action='store', type=int, default=-1,
                        help='Number of workers/processes to run in parallel. (-1 to start one per core.')
    parser.add_argument('-p', '--path', action='store', type=str, default=def_file_path,
                        help='Path to save the results.')

    args = parser.parse_args()

    it = args.iterations

    run_path = path.join(args.path, "run")
    logging_path = path.join(args.path, "log")
    err_path = path.join(args.path, "err")
    launch_file = args.launch_file

    # Static Launch Arguments (Don't change between experiments)
    stat_args = {
        "do_slam": True,
        "do_plots": False,
        "do_error": True,
        "do_coll_err": True,
        "do_gtmap": False,
        "do_odo": False,
        "do_rviz": False,
        "do_gsp_debug": True,
        "sim_pause": False,
        "sim_quiet": True,

        "path_err_coll_path": err_path,
        "path_prefix": run_path
    }

    # Variable Launch Arguments (Different for each experiment)
    # All permutations of these settings will be executed, so be wary of that!
    var_args = OrderedDict([
        ("bag_file", args.moves),
        ("mm", args.map_models),
        ("pw", args.particle_weights),
        ("doPoseImprove", args.pose_improve)
    ])

    # Functions for transforming an argument into its final value and label
    bag_file_format = "Robot_Exp_10Loop_{:03d}m1loc.bag"
    bag_file_format = path.join(slamdata_pck_share, "Simulations", bag_file_format)
    var_args_fun = {
        "val": {
            "bag_file": lambda x: bag_file_format.format(x)
        },
        "lbl": {
            "bag_file": lambda x: "{:03d}mv".format(x),
            "doPoseImprove": lambda x: "sm" if x else "mm"
        }
    }

    # Generate all possible experiment permutations
    keys, values = zip(*var_args.items())
    var_args = [OrderedDict(zip(keys, v)) for v in itertools.product(*values)]

    # Generating final list of arguments by executing their transformation functions if needed.
    experiment_arg_list = []
    for exp_args in var_args:
        experiment_args = {}
        file_prefix = ""
        for k, v in exp_args.items():
            if k in var_args_fun["lbl"]:
                file_prefix += var_args_fun["lbl"][k](v) + "_"
            else:
                file_prefix += str(v) + "_"
            if k in var_args_fun["val"]:
                experiment_args[k] = var_args_fun["val"][k](v)
            else:
                experiment_args[k] = v
        file_prefix = file_prefix[:-1]
        experiment_args["path_err_coll_pfx"] = file_prefix

        tmp_launch_args_dict = stat_args.copy()
        tmp_launch_args_dict.update(experiment_args)
        experiment_arg_list.append(tmp_launch_args_dict)

    # Multiprocess pool settings
    if args.num_workers < 1:
        num_procs = multiprocessing.cpu_count()-1 or 1
    else:
        num_procs = args.num_workers

    multiproc = args.multi_proc
    procs = []
    pool = None

    if multiproc:
        pool = multiprocessing.Pool(processes=num_procs)
        print("Running experiments in a pool of {} processes.".format(num_procs))

    # Main experiments
    for exp_args in experiment_arg_list:
        if not multiproc:
            run_exp_n_times(launchfile_package, launch_file, iterations=it, launch_args_dict=exp_args,
                            log_path=logging_path, monitored_nodes={"any": ["sim", "SLAM/slam"]}, port="auto")
        else:
            proc = pool.apply_async(run_exp_n_times, args=(launchfile_package, launch_file, ),
                                    kwds={"iterations": it,
                                          "launch_args_dict": exp_args,
                                          "log_path": logging_path,
                                          "monitored_nodes": {"all": ["sim", "SLAM/slam"]},
                                          "port": "auto"}
                                    )
            sleep(1.0)
            procs.append(proc)

    if multiproc:
        for proc in procs:
            proc.get()

        pool.close()
        pool.join()

    print("\n" * 10)
    print("*" * 80)
    print("<*>\tALL EXPERIMENTS FINISHED RUNNING")
    print("*" * 80)
