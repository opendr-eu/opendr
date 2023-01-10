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

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

import itertools

from os import path, listdir

from map_simulator.utils import mkdir_p

if __name__ == "__main__":

    import argparse

    FS = ","
    LS = "\n"

    def_data_path = path.join("~", "Desktop", "Experiments", "MethodComparison", "err")
    def_out_path = path.join(def_data_path, "errbar")

    parser = argparse.ArgumentParser(description='Read the data collected into csv files in a given directory \
        and plot error bars.')
    parser.add_argument('-d', '--dir', action='store', type=str, default=def_data_path,
                        help='Path of the directory where the CSV error files are stored.')
    parser.add_argument('-x', '--extension', action='store', type=str, default='csv',
                        help='Data file extension. [Default: csv].')
    parser.add_argument('-o', '--out_dir', action='store', type=str, default=def_out_path,
                        help='Output Directory where histograms will be saved.')

    parser.add_argument('-m', '--max_exps', action='store', type=int, default=0,
                        help='Maximum number of experiments to take into account.')

    args = parser.parse_args()
    data_path = path.expandvars(path.expanduser(args.dir))
    out_path = path.expandvars(path.expanduser(args.out_dir))

    mkdir_p(out_path)

    tmp_ext = ".{}".format(args.extension)
    path_files = listdir(data_path)
    path_files = [f[:f.find(tmp_ext)] for f in path_files if tmp_ext in f]

    path_file_options = [f.split("_") for f in path_files]
    path_file_options = (zip(*path_file_options))
    path_file_options = [sorted(list(set(c))) for c in path_file_options]

    move_options = sorted(map(int, [o[:o.find('mv')] for o in path_file_options[0]]))
    # move_options.remove(20)

    # rem_options = [path_file_options[2]] + path_file_options[4:]
    rem_options = path_file_options[3:]
    rem_combinations = itertools.product(*rem_options)

    curve_options = [path_file_options[1], path_file_options[2]]

    file_headers = ["Moves"] + map(str, move_options)
    file_cols = [file_headers]

    plt.ioff()

    for o1 in rem_combinations:

        fig_lbl = "_".join(o1)

        curves = []
        curve_combinations = itertools.product(*curve_options)

        for o2 in curve_combinations:

            col_lbl = "{}_{}_{}_{}_{}".format(o2[0], o2[1], o1[0], o1[1], o1[2])

            file_means = [col_lbl + "_mean"]
            file_sdevs = [col_lbl + "_sdv"]
            file_sems = [col_lbl + "_sem"]
            file_runs = [col_lbl + "_runs"]

            curve_lbl = "_".join(o2)
            # curve_lbl = str(o2)

            means = []
            sdevs = []
            sems = []
            runs = []

            for m in move_options:
                file_name = "{:03d}mv_{}.{}".format(m, col_lbl, args.extension)

                file_path = path.join(data_path, file_name)

                err = []
                no_experiments = 0

                with open(file_path, 'r') as f:
                    for line in f:
                        if args.max_exps > 0 and args.max_exps <= no_experiments:
                            break

                        try:
                            err.append(float(line.split(FS)[-1]))
                            no_experiments += 1
                        except ValueError:
                            print("Warning: File {} contained unparsable line: {}.".format(file_name, repr(line)))

                means.append(np.mean(err))
                sdevs.append(np.std(err))
                sems.append(sem(err))
                runs.append(no_experiments)

            file_means += means
            file_sdevs += sdevs
            file_sems += sems
            file_runs += runs

            file_cols.append(map(str, file_means))
            file_cols.append(map(str, file_sdevs))
            file_cols.append(map(str, file_sems))
            file_cols.append(map(str, file_runs))

            curves.append((curve_lbl, means, sdevs))

        print("Plotting figure: " + fig_lbl)
        plt.figure()
        for curve in curves:
            plt.errorbar(move_options, curve[1], curve[2], label=curve[0])
        plt.legend()
        plt.xlabel("No. of mapping scans.")
        plt.ylabel("Error (mean +- stddev)")
        plt.xticks(move_options, rotation='vertical')
        plt.title(fig_lbl)
        plt.savefig(path.join(out_path, fig_lbl + "_errbar.svg"))
        plt.close()

    data_file_path = path.join(out_path, "errbar_data.dat")
    lines = zip(*file_cols)

    with open(data_file_path, "w") as f:
        for line in lines:
            f.write(FS.join(line) + LS)
