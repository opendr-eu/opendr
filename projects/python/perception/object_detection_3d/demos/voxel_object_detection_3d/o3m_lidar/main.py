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

from .o3m_lidar import O3MLidar


ip = "0.0.0.0"
port = 42000
buffer_size = 1460


def main():

    lidar = O3MLidar(ip, port, buffer_size, output_mode="point_cloud")

    i = 0

    while True:
        point_cloud = lidar.next()
        if len(point_cloud) > 0:
            print(i, lidar.next()[0])
            i += 1


main()
