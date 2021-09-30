# Copyright 2021 OpenDR European Project
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

from pathlib import Path
from typing import Union


def bump_version(path: Union[str, Path]) -> Path:
    """Bumps the version number for a path if it already exists

    Example::

        bump_version("folder/new_file.json") == Path("folder/new_file.json)
        bump_version("folder/old_file.json") == Path("folder/old_file_1.json)
        bump_version("folder/old_file_1.json") == Path("folder/old_file_2.json)
    """
    if not path.exists():
        return path

    # Check for already bumped versions
    prev_version = None
    try:
        prev_version = max(
            map(
                int,
                filter(
                    lambda s: s.isdigit(),
                    [f.stem.split("_")[-1] for f in path.parent.glob(f"{path.stem}*")],
                ),
            )
        )
        new_version = prev_version + 1
    except ValueError:  # max() arg is an empty sequence
        new_version = 1

    if prev_version and path.stem.endswith(f"_{prev_version}"):
        suffix = f"_{prev_version}"
        new_name = f"{path.stem[:-len(suffix)]}_{new_version}{path.suffix}"
    else:
        new_name = f"{path.stem}_{new_version}{path.suffix}"
    return path.parent / new_name
