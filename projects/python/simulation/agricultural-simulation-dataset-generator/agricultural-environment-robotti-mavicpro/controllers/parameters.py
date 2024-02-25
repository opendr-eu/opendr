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

useRobotti = True  # options: False, True
useMavic = True  # options: False, True
weather_condition = [
        'noon_cloudy_countryside']  # options: noon_cloudy_countryside, dawn_cloudy_empty, noon_stormy_empty, dusk
enable_fog = False  # options: False, True
DATASET_DIR_UAV = '../dataset_location/{}/UAV'.format(weather_condition[0])
DATASET_DIR_ROBOTTI = '../dataset_location/{}/UGV'.format(weather_condition[0])
STOP_ON = 193
MAX_RECORDS_PER_SCENARIO = 19300
OBSTACLES_PER_SCENARIO = 12
WEBOTS_VERSION = "Webots 2023b"
ROBOTTI_MAX_SPEED = 6.28
FIELD_SIZE = [40, 14]
