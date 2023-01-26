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

from ctypes import (
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    Structure,
    c_int8,
    c_float,
)


class PacketHeader(Structure):
    _fields_ = [
        ("Version", c_uint16),
        ("Device", c_uint16),
        ("PacketCounter", c_uint32),
        ("CycleCounter", c_uint32),
        ("NumberOfPacketsInCycle", c_uint16),
        ("IndexOfPacketInCycle", c_uint16),
        ("NumberOfPacketsInChannel", c_uint16),
        ("IndexOfPacketInChannel", c_uint16),
        ("ChannelID", c_uint32),
        ("TotalLengthOfChannel", c_uint32),
        ("LengthPayload", c_uint32),
    ]


class ChannelHeader(Structure):
    _fields_ = [
        ("StartDelimiter", c_uint32),
        ("reserved", c_uint8 * 24),
    ]


class ChannelEnd(Structure):
    _fields_ = [
        ("EndDelimiter", c_uint32),
    ]


class CameraCalibration(Structure):
    _fields_ = [
        ("transX", c_float),
        ("transY", c_float),
        ("transZ", c_float),
        ("rotX", c_float),
        ("rotY", c_float),
        ("rotZ", c_float),
    ]


class FieldOfView(Structure):
    _fields_ = [
        ("upperLeft", c_float * 3),
        ("upperRight", c_float * 3),
        ("lowerLeft", c_float * 3),
        ("lowerRight", c_float * 3),
    ]


class IntrExtrCalib_2d(Structure):
    _fields_ = [
        ("intrCalib_2D_fx", c_float),
        ("intrCalib_2D_fy", c_float),
        ("intrCalib_2D_mx", c_float),
        ("intrCalib_2D_my", c_float),
        ("intrCalib_alpha", c_float),
        ("intrCalib_k1", c_float),
        ("intrCalib_k2", c_float),
        ("intrCalib_k5", c_float),
        ("intrCalib_k3", c_float),
        ("intrCalib_k4", c_float),
        ("extrCalib_center_tx", c_float),
        ("extrCalib_center_ty", c_float),
        ("extrCalib_center_tz", c_float),
        ("extrCalib_delta_tx", c_float),
        ("extrCalib_delta_ty", c_float),
        ("extrCalib_delta_tz", c_float),
        ("extrCalib_rot_x", c_float),
        ("extrCalib_rot_y", c_float),
        ("extrCalib_rot_z", c_float),
    ]


class DistanceImageResult(Structure):
    _fields_ = [
        ("sensorWidth", c_uint16),  # 12
        ("sensorHeight", c_uint16),
        ("distanceData", c_uint16 * 1024),
        ("X", c_float * 1024),
        ("Y", c_float * 1024),
        ("Z", c_float * 1024),
        ("confidence", c_uint16 * 1024),
        ("amplitude", c_uint16 * 1024),
        ("amplitude_normalization", c_float * 4),
        ("masterclockTimestamp", c_uint32),
        ("frameCounter", c_uint32),
        ("available", c_uint32),
        ("cameraCalibration", CameraCalibration),
        ("fieldOfView", FieldOfView),
        ("intrExtrCalib_2d", IntrExtrCalib_2d),
        ("illuPosition", c_float * 3),
        ("blockageRatio", c_float),
        ("blockageAvailable", c_uint8),
        ("pad_001", c_uint8),
        ("pad_002", c_uint8),
        ("pad_003", c_uint8),
    ]


class CommonCalibrationResultData(Structure):
    _fields_ = [
        ("transX", c_float),
        ("transY", c_float),
        ("transZ", c_float),
        ("rotX", c_float),
        ("rotY", c_float),
        ("rotZ", c_float),
    ]


class CommonCalibrationResult(Structure):
    _fields_ = [
        ("calibValid", c_int32),
        ("calibrationStableCounter", c_int32),
        ("calibResult", CommonCalibrationResultData),
    ]


class TriangleDetections(Structure):
    _fields_ = [
        ("score", c_float),
        ("pos3D", c_float * 3),
        ("corners", c_float * 3 * 2),
    ]


class PacCalibrationResult(Structure):
    _fields_ = [
        ("numTrianglesDetected", c_uint8),
        ("pad_001", c_uint8),
        ("pad_002", c_uint8),
        ("pad_003", c_uint8),
        ("triangleDetections", TriangleDetections * 8),
        ("frameValid", c_int32),
        ("frameReprojectError", c_float),
    ]


class PlaneEstimation(Structure):
    _fields_ = [
        ("pitchAngle", c_float),
        ("rollAngle", c_float),
        ("camHeight", c_float),
        ("normalx", c_float),
        ("normaly", c_float),
        ("normalz", c_float),
    ]


class StreetCalibrationResult(Structure):
    _fields_ = [
        ("planeValid", c_int32),
        ("planeEstimation", PlaneEstimation),
        ("plausibility", c_float),
        ("distanceDeviation", c_float),
    ]


class CalibrationResult(Structure):
    _fields_ = [
        ("commonCalibrationResult", CommonCalibrationResult),
        ("pacCalibrationResult", PacCalibrationResult),
        ("streetCalibrationResult", StreetCalibrationResult),
    ]


class LogicOutput(Structure):
    _fields_ = [
        ("digitalOutput", c_uint8 * 100),
        ("analogOutput", c_float * 20),
    ]


class Channel8Data(Structure):
    _fields_ = [
        ("magic_no", c_int8 * 4),
        ("struct_id", c_int8 * 4),
        ("version", c_uint8 * 2),
        ("pad_001", c_uint8),  # 10
        ("pad_002", c_uint8),  # 11
        ("distanceImageResult", DistanceImageResult),
        ("calibrationResult", CalibrationResult),
        ("logicOutput", LogicOutput),
    ]
