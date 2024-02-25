#!/usr/bin/env python3

# Copyright 2020-2024 OpenDR European Project
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


# Loop through the src folder and generate the dependencies list

import os
import sys
from configparser import ConfigParser
from warnings import warn

global flag_efficientNet
flag_efficientNet = ''

python_prerequisites_file = "python_prerequisites.txt"
python_file = "python_dependencies.txt"
linux_file = "linux_dependencies.txt"


def read_ini(path):
    opendr_device = os.environ.get('OPENDR_DEVICE', default='cpu')
    parser = ConfigParser()
    parser.read(path)

    def read_ini_key(key, summary_file):
        tool_device = parser.get('device', 'opendr_device', fallback='cpu')
        if opendr_device == 'cpu' and tool_device == 'gpu':
            return
        if parser.has_option(section, key):
            dependencies = parser.get(section, key)
            if dependencies:
                for package in dependencies.split('\n'):
                    if 'efficientNet' in package:
                        efficientNetTrick(package)
                        continue
                    with open(summary_file, "a") as f:
                        f.write(os.path.expandvars(package) + '\n')

    read_ini_key('python-dependencies', python_prerequisites_file)
    read_ini_key('python', python_file)
    read_ini_key('linux', linux_file)


def efficientNetTrick(package):
    global flag_efficientNet
    if 'EfficientLPS' in package:
        flag_efficientNet = package
    # EfficientPS works with both versions of efficientNet but EfficientLPS works 
    # only with EfficientLPS version
    elif 'EfficientPS' in package and 'EfficientLPS' not in flag_efficientNet:
        flag_efficientNet = package


def verbose_print(message, verbose):
    if verbose:
        print(message)


def add_dep_to_dict(dep_dict, dep, new_constraint, verbose=False):
    if dep not in dep_dict.keys():
        dep_dict[dep] = new_constraint
        verbose_print(f"no issues adding new dependency: {dep}, {new_constraint}", verbose)
        return
    elif dep in dep_dict.keys() and dep_dict[dep] != new_constraint:
        verbose_print(f"key already exists: {dep} as {dep_dict[dep]}, " +
                      f"trying to re-add with new_constraint {new_constraint}", verbose)

        # Cases where old/new constraint is loose/strict(==) are clear
        if new_constraint is None:
            verbose_print("Original is strict, not re-adding as loose, skipping...", verbose)
        elif dep_dict[dep] is None:
            verbose_print("Original is loose, re-adding with more strict, adding...", verbose)
            dep_dict[dep] = new_constraint

        # Cases where one of the dependencies is strict and the other is half-strict (>, <, >=, <=) are not that clear
        elif "==" in new_constraint and "==" not in dep_dict[dep]:
            verbose_print("Original is half-strict, re-adding with strict, adding...", verbose)
            if dep_dict[dep][0] != new_constraint[0]:
                warn("[WARNING] Filtering dependencies, probable clash of strict and non-strict versions "
                     f"of same package. {dep} tries to constraint both to version {dep_dict[dep]} and "
                     f"{new_constraint}, please review.")
            dep_dict[dep] = new_constraint
        elif "==" in dep_dict[dep] and "==" not in new_constraint:
            verbose_print("Original is strict (==), new isn't, skipping...", verbose)
            if dep_dict[dep][0] != new_constraint[0]:
                warn("[WARNING] Filtering dependencies, probable clash of strict and non-strict versions "
                     f"of same package. {dep} tries to constraint both to version {dep_dict[dep]} and "
                     f"{new_constraint}, please review.")

        # Critical clash of strict but different constraints
        elif "==" in dep_dict[dep] and "==" in new_constraint:
            raise EnvironmentError(f"Filtering dependencies, critical clash of strict versions of same package. "
                                   f"{dep} tries to constraint both to version {dep_dict[dep]} and {new_constraint}.")

        # Remaining cases with half-strict constraints, not that clear
        else:
            warn("[WARNING] Probably case where different half-strict (<=, >=, etc.) constraints clash. "
                 f"Attempted to add {dep}, {new_constraint}. Kept original dependency {dep}, {dep_dict[dep]}, "
                 "please review.")
        return
    else:
        verbose_print(f"key already exists: {dep}, as  {dep_dict[dep]}, " +
                      f"tried to add with same constraint {new_constraint}", verbose)
        return


def filter_dependencies(dependencies, verbose=False):
    filt_dependencies = {}
    filtered_dependencies_list = []

    # Filter dependencies by adding them in a dictionary
    for d in dependencies:
        if "git" in d:
            # Git dependency, add as-is
            add_dep_to_dict(filt_dependencies, d, None, verbose)
        elif "==" in d:
            d, dep_ver = d.split("==")[0], d.split("==")[1]
            add_dep_to_dict(filt_dependencies, d, [dep_ver, "=="], verbose)
        elif "," in d:
            # Upper and lower constraint, add as-is
            add_dep_to_dict(filt_dependencies, d, None, verbose)
            warn(f"[WARNING] added a dependency {d}, with both an upper and a lower version, please review.")
        elif "<=" in d:
            d, dep_ver = d.split("<=")[0], d.split("<=")[1]
            add_dep_to_dict(filt_dependencies, d, [dep_ver, "<="], verbose)
        elif ">=" in d:
            d, dep_ver = d.split(">=")[0], d.split(">=")[1]
            add_dep_to_dict(filt_dependencies, d, [dep_ver, ">="], verbose)
        elif ">" in d:
            d, dep_ver = d.split(">")[0], d.split(">")[1]
            add_dep_to_dict(filt_dependencies, d, [dep_ver, ">"], verbose)
        elif "<" in d:
            d, dep_ver = d.split("<")[0], d.split("<")[1]
            add_dep_to_dict(filt_dependencies, d, [dep_ver, "<"], verbose)
        else:
            # Simple dependency (no constraint), add as-is
            add_dep_to_dict(filt_dependencies, d, None, verbose)
        if verbose:
            print("")

    # Gather dependencies with their constraints in a list
    for key, val in filt_dependencies.items():
        full_dep = key
        if val is not None:
            full_dep += val[1] + val[0]
        filtered_dependencies_list.append(full_dep)
    if filtered_dependencies_list[-1] == "":
        filtered_dependencies_list = filtered_dependencies_list[0:-1]  # Remove empty last line

    return filtered_dependencies_list


# Parse arguments
section = "runtime"
if len(sys.argv) > 1:
    section = sys.argv[1]
if section not in ["runtime", "compilation"]:
    sys.exit("Invalid dependencies type: " + section + ".\nExpected: [runtime, compilation]")
global_dependencies = True if '--global' in sys.argv else False

# Clear dependencies
if os.path.exists(python_prerequisites_file):
    os.remove(python_prerequisites_file)
if os.path.exists(python_file):
    os.remove(python_file)
if os.path.exists(linux_file):
    os.remove(linux_file)

# Extract generic dependencies
read_ini('dependencies.ini')
# Loop through tools and extract dependencies
if not global_dependencies:
    opendr_home = os.environ.get('OPENDR_HOME')
    for dir_to_walk in ['src', 'projects/python/control/eagerx']:
        for subdir, dirs, files in os.walk(os.path.join(opendr_home, dir_to_walk)):
            for filename in files:
                if filename == 'dependencies.ini':
                    read_ini(os.path.join(subdir, filename))
with open(python_file, "a") as f:
    f.write(os.path.expandvars(flag_efficientNet) + '\n')

# Filter python dependencies
python_dependencies = []
with open(python_file, "r") as f:
    for line in f:
        # Grab line and remove newline char
        python_dependencies.append(line.replace("\n", ""))

# Set verbose to True to debug dependencies in detail
filtered_dependencies = filter_dependencies(python_dependencies, verbose=False)
print(f"{len(python_dependencies)} dependencies filtered down to {len(filtered_dependencies)}")

# # Make igibson last in the installation order, which can fix possible igibson installation error
# filtered_dependencies.remove("igibson==2.0.3")
# filtered_dependencies.append("igibson==2.0.3")

with open(python_file, "w") as f:
    for i in range(len(filtered_dependencies)):
        if i < len(filtered_dependencies) - 1:
            f.write(filtered_dependencies[i] + "\n")
        else:
            f.write(filtered_dependencies[i])
