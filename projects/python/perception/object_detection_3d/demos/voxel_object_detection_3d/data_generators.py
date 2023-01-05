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

from opendr.engine.datasets import PointCloudsDatasetIterator


def disk_point_cloud_generator(
    path, num_point_features=4, cycle=True, count=None
):
    dataset = PointCloudsDatasetIterator(
        path, num_point_features=num_point_features
    )

    i = 0

    len_dataset = len(dataset) if count is None else count

    while i < len_dataset or cycle:
        yield dataset[i % len_dataset]
        i += 1


def lidar_point_cloud_generator(lidar):

    while True:
        yield lidar.next()
