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

import argparse
import rospkg
import os

from map_simulator.map_simulator_2d import MapSimulator2D

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate a ROSbag file from a simulated robot trajectory.")

    parser.add_argument('-i', '--input', action='store', help='Input JSON robot config file', type=str, required=True)
    parser.add_argument('-o', '--output', action='store', help='Output ROSbag file', type=str, required=False)

    parser.add_argument('-p', '--preview', action='store_true')
    parser.add_argument('-s', '--search_paths', action='store', type=str,
                        default='',
                        help='Search paths for the input and include files separated by colons (:)')

    args, override_args = parser.parse_known_args()

    r = rospkg.RosPack()
    pck_share = r.get_path('map_simulator')
    subdirs = [['scenarios'], ['scenarios', 'robots'], ['scenarios', 'maps'],
               ['scenarios', 'sensors'], ['scenarios', 'commands']]
    search_dirs = ['.']
    for s in subdirs:
        path = ""
        for ss in s:
            path = os.path.join(path, ss)

        path = os.path.join(pck_share, path)
        search_dirs.append(path)

    arg_search_dirs = []
    if args.search_paths is not None and args.search_paths != "":
        tmp_search_dirs = args.search_paths.split(':')

        for p in tmp_search_dirs:
            if p:
                arg_search_dirs.append(os.path.realpath(p))

    arg_search_dirs.extend(search_dirs)

    override_str = None

    if len(override_args) > 0:
        override_str = '{'
        for arg in override_args:
            arg_keyval = arg.split(":=")
            override_str += '"' + str(arg_keyval[0]) + '": "' + str(arg_keyval[1]) + '",'

        override_str = override_str[0:-1] + "}"

    print("OVERRIDE_STR:   ", override_str)

    simulator = MapSimulator2D(args.input, arg_search_dirs, override_params=override_str)
    simulator.simulate(args.output, display=args.preview)
