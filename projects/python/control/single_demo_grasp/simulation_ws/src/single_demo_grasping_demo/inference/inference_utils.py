#!/usr/bin/env python

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


import math
import pandas as pd
import numpy as np
from collections import Counter


def correct_orientation_ref(angle):

    if angle <= 90:
        angle += 90
    if angle > 90:
        angle += -270

    return angle


def get_angle(input_kps, mode):
    kps_x_list = input_kps[:, 0]
    kps_y_list = input_kps[:, 1]
    kps_x_list = kps_x_list[::-1]
    kps_y_list = kps_y_list[::-1]

    d = {'X': kps_x_list,
         'Y': kps_y_list}

    df = pd.DataFrame(data=d)
    # move the origin
    x = (df["X"] - df["X"][0])
    y = (df["Y"] - df["Y"][0])

    if mode == 1:
        list_xy = (np.arctan2(y, x) * 180 / math.pi).astype(int)
        occurence_count = Counter(list_xy)
        return occurence_count.most_common(1)[0][0]
    else:
        x = np.mean(x)
        y = np.mean(y)
        return np.arctan2(y, x) * 180 / math.pi


def get_kps_center(input_kps):

    kps_x_list = input_kps[:, 0]
    kps_y_list = input_kps[:, 1]
    kps_x_list = kps_x_list[::-1]
    kps_y_list = kps_y_list[::-1]
    d = {'X': kps_x_list,
         'Y': kps_y_list}
    df = pd.DataFrame(data=d)

    x = np.mean(df["X"])
    y = np.mean(df["Y"])
    return [x, y]
