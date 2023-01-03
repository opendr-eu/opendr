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

import ctypes
from .structures import PacketHeader, ChannelHeader, ChannelEnd, Channel8Data


class Channel:
    def __init__(self, id=8) -> None:
        self.target_channel_id = id

        assert self.target_channel_id in [8]

        self.previous_packet_counter_valid = True
        self.previous_packet_counter = 0
        self.start_of_channel_found = False
        self.position_in_channel = 0
        self.buffer = None

    def update(self, packet):
        packet_header = PacketHeader.from_buffer_copy(packet, 0)
        self.__check_packet_counter(packet_header)

        if self.target_channel_id == packet_header.ChannelID:
            self.__check_start_of_channel(packet_header)

            if self.start_of_channel_found:
                self.__process_packet(packet, packet_header)

                if (
                    packet_header.IndexOfPacketInChannel ==
                    packet_header.NumberOfPacketsInChannel - 1
                ):
                    return self.__process_channel()

        return None

    def __process_packet(self, packet, packet_header):
        start = ctypes.sizeof(packet_header)
        length = len(packet) - start

        if packet_header.IndexOfPacketInChannel == 0:
            start += ctypes.sizeof(ChannelHeader)
            length -= ctypes.sizeof(ChannelHeader)

        if (
            packet_header.IndexOfPacketInChannel ==
            packet_header.NumberOfPacketsInChannel - 1
        ):
            length -= ctypes.sizeof(ChannelEnd)

        if self.position_in_channel + length > len(self.buffer):
            raise Exception("Channel buffer is too small")

        self.buffer[
            self.position_in_channel: self.position_in_channel + length
        ] = packet[start: start + length]
        self.position_in_channel += length

    def __process_channel(self):

        data = Channel8Data.from_buffer_copy(self.buffer)

        if not self.__check_magic_no_and_version(data):
            raise Exception("Wrong magic_no and version")

        if ctypes.sizeof(data) != 19224 and self.target_channel_id == 8:
            raise Exception("Wrong size for Channel:", self.target_channel_id)

        return data

    def __check_magic_no_and_version(self, data: Channel8Data):

        is_o3m = bytes(data.magic_no).decode("ascii") == "O3M!"
        is_dia1 = bytes(data.struct_id).decode("ascii") == "DIA1"
        is_1_4 = bytes(data.version) == b"\x01\x04"

        return is_o3m and is_dia1 and is_1_4

    def __check_packet_counter(self, packet_header):
        if self.previous_packet_counter_valid and (
            packet_header.PacketCounter - self.previous_packet_counter != 1
        ):

            if self.previous_packet_counter != 0:
                print(
                    "Packet counter jumped from",
                    self.previous_packet_counter,
                    "to",
                    packet_header.PacketCounter,
                )
            self.start_of_channel_found = False

        self.previous_packet_counter = packet_header.PacketCounter
        self.previous_packet_counter_valid = True

    def __check_start_of_channel(self, packet_header):
        if packet_header.IndexOfPacketInChannel == 0:
            self.start_of_channel_found = True

            # if self.buffer is None:
            self.buffer = bytearray(packet_header.TotalLengthOfChannel)
            self.position_in_channel = 0
