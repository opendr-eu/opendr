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

import socket
from .structures import Channel8Data
from .channel import Channel
from opendr.engine.data import PointCloud
import threading
import numpy as np


def point_cloud_from_channel_8_data(data: Channel8Data):

    distance_image = data.distanceImageResult
    positions = np.empty((len(distance_image.X), 4))

    positions[:, 0] = distance_image.X
    positions[:, 1] = distance_image.Y
    positions[:, 2] = distance_image.Z
    positions[:, 3] = 1

    return positions


class O3MLidar:
    def __init__(
        self,
        ip,
        port,
        buffer_size=1460,
        channel_id=8,
        output_mode="point_cloud",
    ) -> None:
        self.socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        self.socket.bind((ip, port))
        print("UDP socket", ip, port)

        self.channel = Channel(channel_id)
        self.buffer_size = buffer_size
        self.output_mode = output_mode

        self.iterate_thread = threading.Thread(target=self.__iterate_scans)
        self.iterate_thread.start()
        self.lock = threading.Lock()
        self.last_output = np.zeros((0, 4), dtype=np.float32)
        self.running = True

    def __iterate_scans(self):

        while self.running:
            with self.lock:
                packet, _ = self.socket.recvfrom(self.buffer_size)
                result = self.channel.update(packet)
                if result is not None:
                    if self.output_mode == "channel_8":
                        self.last_output = result
                    elif self.output_mode == "point_cloud":
                        self.last_output = point_cloud_from_channel_8_data(
                            result
                        )
                    else:
                        raise Exception(
                            "Unknown output_mode:", self.output_mode
                        )

    def next(self):
        with self.lock:
            return PointCloud(self.last_output)

    def stop(self):
        self.running = False
        self.iterate_thread.join()
