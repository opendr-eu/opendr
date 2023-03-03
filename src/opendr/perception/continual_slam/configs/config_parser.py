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

import dataclasses
import sys
from os import PathLike
from pathlib import Path
from typing import List, Union, get_args, get_origin

import yaml

from opendr.perception.continual_slam.datasets import Config as Dataset
from opendr.perception.continual_slam.algorithm.parsing.config import Config as DepthPosePrediction
from opendr.perception.continual_slam.algorithm.loop_closure.config import LoopClosureDetection

class ConfigParser():
    def __init__(self, config_file: Union[str, PathLike, Path]) -> None:
        self.filename = Path(config_file)

        self.config_dict = {}
        self.dataset = None
        self.depth_pose = None
        self.loop_closure = None

        self.parse()

    def parse(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.config_dict = yaml.safe_load(file)

        # Read lists as tuples
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if isinstance(value, List):
                    config_type[key] = tuple(value)

        # Correct wrongly parsed data types
        for config_type_key, config_type in self.config_dict.items():
            config_type_class = getattr(sys.modules[__name__], config_type_key)
            for field in dataclasses.fields(config_type_class):
                if field.name == 'config_file':
                    continue
                value = config_type[field.name]
                if value is not None:
                    expected_type = field.type
                    if get_origin(field.type) is not None:
                        expected_type = get_origin(field.type)
                    if expected_type is Union:
                        expected_type = []
                        for tp in get_args(field.type):
                            if get_origin(tp) is not None:
                                expected_type.append(get_origin(tp))
                            else:
                                expected_type.append(tp)
                    else:
                        expected_type = [expected_type]

                    # Remove the NoneType before attempting conversions
                    expected_type = [tp for tp in expected_type if tp is not type(None)]

                    if not any(isinstance(value, tp) for tp in expected_type):
                        if len(expected_type) == 1:
                            print(f'[CONFIG] Converting {field.name} from {type(value).__name__} '
                                  f'to {expected_type[0].__name__}.')
                            config_type[field.name] = expected_type[0](value)
                        else:
                            assert False, 'Found an unknown issue!'
                elif get_origin(field.type) is Union and type(None) in get_args(field.type):
                    # Is optional
                    print(f'[CONFIG] Setting {field.name} to None.')
                elif get_origin(field.type) is Union:
                    # Is required
                    raise ValueError(f'[CONFIG] Required parameter missing: {field.name}')
                else:
                    assert False, 'Found an unknown issue!'

        # Add the path to the config file to all Configurations
        for config_type in self.config_dict.values():
            config_type['config_file'] = self.filename

        # Convert paths to absolute paths
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if isinstance(value, Path):
                    config_type[key] = value.absolute()

        # Read the sections
        if 'Dataset' in self.config_dict:
            self.dataset = Dataset(**self.config_dict['Dataset'])
        if 'DepthPosePrediction' in self.config_dict:
            self.depth_pose = DepthPosePrediction(**self.config_dict['DepthPosePrediction'])
        if 'LoopClosureDetection' in self.config_dict:
            self.loop_closure = LoopClosureDetection(**self.config_dict['LoopClosureDetection'])

    def __str__(self):
        string = ''
        for config_type_name, config_type in self.config_dict.items():
            string += f'----- {config_type_name} --- START -----\n'
            for name, value in config_type.items():
                if name != 'config_file':
                    string += f'{name:25} : {value}\n'
            string += f'----- {config_type_name} --- END -------\n'
        string = string[:-1]
        return string
