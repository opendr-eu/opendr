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
from pip._internal import main as pip
from configparser import ConfigParser

python_file = "python_dependencies.txt"
linux_runtime_file = "linux_runtime_dependencies.txt"

def read_ini(path):
    parser = ConfigParser()
    parser.read(path)
    if parser.has_option('runtime', 'python'):
        python_dependencies = parser.get('runtime', 'python')
        if python_dependencies:
            for package in python_dependencies.split():
                f = open(python_file, "a")
                f.write(package + '\n')
    if parser.has_option('runtime', 'linux'):
        linux_dependencies = parser.get('runtime', 'linux')
        if linux_dependencies:
            for package in linux_dependencies.split():
                f = open(linux_runtime_file, "a")
                f.write(package + '\n')


# Clear dependencies
open(python_file, 'w').close()
open(linux_runtime_file, 'w').close()
# Extract generic dependencies
read_ini('dependencies.ini')
# Loop through tools and extract dependencies
opendr_home = os.environ.get('OPENDR_HOME')
for subdir, dirs, files in os.walk(opendr_home + '/src'):
    for filename in files:
        if filename == 'dependencies.ini':
            read_ini(subdir + os.sep + filename)
