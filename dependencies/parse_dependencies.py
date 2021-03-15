#!/usr/bin/env python3

# Copyright 2020-2021 OpenDR European Project
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

python_file = "python_dependencies.txt"
linux_file = "linux_dependencies.txt"


def read_ini(path):
    parser = ConfigParser()
    parser.read(path)
    if parser.has_option(section, 'python'):
        python_dependencies = parser.get(section, 'python')
        if python_dependencies:
            for package in python_dependencies.split():
                f = open(python_file, "a")
                f.write(package + '\n')
    if parser.has_option(section, 'linux'):
        linux_dependencies = parser.get(section, 'linux')
        if linux_dependencies:
            for package in linux_dependencies.split():
                f = open(linux_file, "a")
                f.write(package + '\n')


# Parse arguments
section = "runtime"
if len(sys.argv) > 1:
    section = sys.argv[1]
if section not in ["runtime", "compilation"]:
    sys.exit("Invalid dependencies type: " + section + ".\nExpected: [runtime, compilation]")

# Clear dependencies
if os.path.exists(python_file):
    os.remove(python_file)
if os.path.exists(linux_file):
    os.remove(linux_file)

# Extract generic dependencies
read_ini('dependencies.ini')
# Loop through tools and extract dependencies
opendr_home = os.environ.get('OPENDR_HOME')
for subdir, dirs, files in os.walk(opendr_home + '/src'):
    for filename in files:
        if filename == 'dependencies.ini':
            read_ini(subdir + os.sep + filename)
