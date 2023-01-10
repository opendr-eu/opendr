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

from urllib.request import urlretrieve
from opendr.engine.constants import OPENDR_SERVER_URL

url = OPENDR_SERVER_URL + "planning/end_to_end_planning/ardupilot.zip"
file_destination = "./ardupilot.zip"
urlretrieve(url=url, filename=file_destination)
