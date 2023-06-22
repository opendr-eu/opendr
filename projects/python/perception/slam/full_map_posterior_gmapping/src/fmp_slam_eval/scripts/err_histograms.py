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

from os import path, listdir

from itertools import product

from map_simulator.utils import mkdir_p


def comma_list(list_string):
    return list_string.strip.split(',')


if __name__ == "__main__":

    import argparse

    def_data_path = path.join("~", "Desktop", "Experiments", "MethodComparison", "err")
    def_out_path = path.join(def_data_path, "hist")

    file_filters = {
        'n_moves':  {'arg_abr': 'n', 'desc': 'number of moves',            'index': 0, 'values': []},
        'm_model':  {'arg_abr': 'm', 'desc': 'map models',                 'index': 1, 'values': []},
        'p_weight': {'arg_abr': 'w', 'desc': 'particle weighting methods', 'index': 2, 'values': []},
        'imp_pose': {'arg_abr': 'i', 'desc': 'pose improve methods',       'index': 3, 'values': []},
        'err_type': {'arg_abr': 'e', 'desc': 'error types',                'index': 4, 'values': []},
        'test_env': {'arg_abr': 't', 'desc': 'test environments',          'index': 5, 'values': []}
    }

    chc_file_filters = sorted(list(file_filters.keys()), key=lambda x: file_filters[x]['index'])

    parser = argparse.ArgumentParser(description='Read the data collected into csv files in a given directory \
        and plot histograms.')
    parser.add_argument('-d', '--dir', action='store', type=str, default=def_data_path,
                        help='Path of the directory where the CSV error files are stored.')
    parser.add_argument('-x', '--extension', action='store', type=str, default='csv',
                        help='Data file extension. [Default: csv].')
    parser.add_argument('-o', '--out_dir', action='store', type=str, default=def_out_path,
                        help='Output Directory where histograms will be saved.')

    # Combine histograms in same plot by this field
    parser.add_argument('-c', '--combine_by', action='store', type=str, choices=chc_file_filters,
                        default='m_model', help='Combine histograms into same plot by field. [Default: "map_model".]')

    # Filter files by
    for file_filter in chc_file_filters:
        file_filter_dict = file_filters[file_filter]
        short_optn = '-{}'.format(file_filter_dict['arg_abr'])
        long_optn = '--{}'.format(file_filter)
        desc = 'Comma separated list of {desc}. If None, then all different {desc}' \
               'in the directory will be used. [Default None].'.format(desc=file_filter_dict['desc'])
        parser.add_argument(short_optn, long_optn, action='store', type=comma_list, default=None, help=desc)

    args = parser.parse_args()
    arg_dict = vars(args)
    data_path = path.expandvars(path.expanduser(args.dir))
    out_path = path.expandvars(path.expanduser(args.out_dir))

    # Available data options in data path
    avl_options = {}

    # If any of the filters was set to None, then check the available types from the files in the path.
    get_types_from_file = False
    for file_filter in chc_file_filters:
        if arg_dict[file_filter] is None:
            get_types_from_file = True
            break
    if get_types_from_file:
        tmp_ext = ".{}".format(args.extension)
        path_files = listdir(data_path)
        path_files = [f[:f.find(tmp_ext)] for f in path_files if tmp_ext in f]

        path_files = [f.split("_") for f in path_files]
        path_files = (zip(*path_files))
        path_files = [set(c) for c in path_files]

        for file_filter in chc_file_filters:
            avl_options[file_filter] = path_files[file_filters[file_filter]['index']]

    for file_filter in chc_file_filters:
        if arg_dict[file_filter] is None:
            file_filters[file_filter]['values'] = sorted(list(avl_options[file_filter]))
        else:
            file_filters[file_filter]['values'] = sorted(list(arg_dict[file_filter]))

    if not path.exists(out_path):
        mkdir_p(out_path)

    FS = ","
    LS = "\n"
    extension = "csv"

    data_header = ["Experiment_Num"]
    data = []
    max_experiments = -1

    plt.ioff()

    combination_filters = sorted(file_filters.keys(), key=lambda x: file_filters[x]['index'])
    combination_filters.remove(args.combine_by)

    file_combinations = product(*[file_filters[k]['values'] for k in combination_filters])
    file_combination_orders = [file_filters[k]['index'] for k in combination_filters]
    file_combination_orders.append(file_filters[args.combine_by]['index'])
    combine_options = file_filters[args.combine_by]['values']

    print("Saving figures to: " + out_path)
    for file_comb in file_combinations:

        hist_plots = []

        for combine_by in combine_options:

            file_name_list = [f for f in file_comb]
            file_name_list.append(combine_by)
            file_name_list = sorted(zip(file_combination_orders, file_name_list))
            file_name_list = [f[1] for f in file_name_list]

            file_name = "_".join(file_name_list)

            err_file_path = path.join(data_path, file_name + '.' + extension)
            data_header.append(file_name)
            err = []
            no_experiments = 0
            with open(err_file_path, 'r') as f:
                for line in f:
                    err.append(float(line.split(FS)[-1]))
                    no_experiments += 1

            hist_plots.append({'lbl': combine_by, 'dat': err, 'cnt': no_experiments})
            max_experiments = max(max_experiments, no_experiments)
            data.append(err)

        fig_name = "_".join(file_comb)
        print("Plotting figure: " + fig_name)
        plt.figure()
        err_data = [e['dat'] for e in hist_plots]
        bins = np.histogram(np.hstack(err_data), bins=20)[1]
        for hist in hist_plots:
            label = '{} ({} exp.)'.format(hist['lbl'], hist['cnt'])
            plt.hist(hist['dat'], bins, label=label, density=True, alpha=1.0/len(combine_options))
        plt.legend()
        plt.title(fig_name)
        plt.savefig(path.join(out_path, fig_name + "_hist.svg"))
        plt.close()

    data_file_path = path.join(out_path, "hist_data.dat")
    print("Saving data file to: " + data_file_path)

    lines = [FS.join(data_header)]

    mean_line = ["Means"]
    var_line = ["Variances"]
    sem_line = ["Standard_Error_of_the_Mean"]

    for e_data in data:
        mean_line.append(str(np.mean(e_data)))
        var_line.append(str(np.var(e_data)))
        sem_line.append(str(sem(e_data)))

    lines.append(FS.join(mean_line))
    lines.append(FS.join(var_line))
    lines.append(FS.join(sem_line))

    for i in range(max_experiments):
        line = [str(i + 1)]

        for exp_data in data:
            if i < len(exp_data):
                value = str(exp_data[i])
            else:
                value = ""
            line.append(value)

        lines.append(FS.join(line))

    with open(data_file_path, "w") as f:
        for line in lines:
            f.write(line + LS)
