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
from typing import Optional
from opendr.engine.data import Image as BaseImage


class Image(BaseImage):
    def __init__(self, data=None, filename: Optional[str] = None, dtype=np.uint8):
        super().__init__(data, dtype)
        self._filename = filename

    @property
    def filename(self) -> str:
        if self._filename is None:
            raise ValueError("Filename is not set.")
        return self._filename
