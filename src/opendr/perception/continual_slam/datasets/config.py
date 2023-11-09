# flake8: noqa
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
from pathlib import Path
from typing import Optional, Tuple


@dataclasses.dataclass
class Dataset:
    dataset: str
    config_file: Path
    dataset_path: Optional[Path]
    scales: Optional[Tuple[int, ...]]
    height: Optional[int]
    width: Optional[int]
    frame_ids: Tuple[int, ...]
