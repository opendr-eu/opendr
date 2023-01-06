#!/usr/bin/env python3

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


# Loop through the src folder and generate the dependencies list

import os
import sys
from configparser import ConfigParser

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
                    with open(summary_file, "a") as f:
                        f.write(os.path.expandvars(package) + '\n')
    read_ini_key('python-dependencies', python_prerequisites_file)
    read_ini_key('python', python_file)
    read_ini_key('linux', linux_file)


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
