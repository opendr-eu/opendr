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

import inspect
import os
from configparser import ConfigParser
from pathlib import Path


def is_unsupported_device(_class):
    opendr_device = os.environ.get('OPENDR_DEVICE', default='cpu')
    tool_device = get_tool_device(_class, default='cpu')
    if opendr_device == 'cpu' and tool_device == 'gpu':
        return True
    return False


def get_tool_device(_class, default: str = 'cpu'):
    learner_src_file = Path(inspect.getfile(_class))
    dependencies_file = learner_src_file.parent / 'dependencies.ini'
    if not dependencies_file.exists():
        raise FileNotFoundError(f'No "dependencies.ini" file found for {_class}')
    parser = ConfigParser()
    parser.read(dependencies_file)
    tool_device = parser.get('device', 'opendr_device')
    return tool_device
