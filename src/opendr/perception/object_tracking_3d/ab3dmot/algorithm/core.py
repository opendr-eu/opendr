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

from typing import List
import numba
import copy
import numpy as np
from scipy.spatial import ConvexHull


@numba.jit
def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@numba.jit
def corner_box3d_volume(corners: np.array):  # [8, 3] -> []

    result = (
        np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2)) *
        np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2)) *
        np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    )
    return result


def polygon_clip(subject_polygon, clip_polygon):  # [(x, y)] -> [(x, y)] -> [(x, y))
    def is_inside(p, clip_polygon1, clip_polygon2):
        return (clip_polygon2[0] - clip_polygon1[0]) * (p[1] - clip_polygon1[1]) > (
            clip_polygon2[1] - clip_polygon1[1]
        ) * (p[0] - clip_polygon1[0])

    def intersection(clip_polygon1, clip_polygon2):
        dc = [clip_polygon1[0] - clip_polygon2[0], clip_polygon1[1] - clip_polygon2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = clip_polygon1[0] * clip_polygon2[1] - clip_polygon1[1] * clip_polygon2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subject_polygon
    cp1 = clip_polygon[-1]

    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if is_inside(e, cp1, cp2):
                if not is_inside(s, cp1, cp2):
                    outputList.append(intersection(cp1, cp2))
                outputList.append(e)
            elif is_inside(s, cp1, cp2):
                outputList.append(intersection(cp1, cp2))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


@numba.jit
def convex_hull_intersection(
    polygon1: List[tuple], polygon2: List[tuple]
):  # [(x, y)] -> [(x, y)] -> ([(x, y), []])
    inter_p = polygon_clip(polygon1, polygon2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def iou3D(corners1, corners2):  # [8, 3] -> [8, 3] -> ([], [])
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = polygon_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = polygon_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    _, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    y_max = min(corners1[0, 1], corners2[0, 1])
    y_min = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, y_max - y_min)
    vol1 = corner_box3d_volume(corners1)
    vol2 = corner_box3d_volume(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


@numba.jit
def rotation_matrix_y(t):  # [] -> [3, 3]
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):  # [7] -> [8, 3]
    bbox3d = copy.copy(bbox3d_input)
    rot_matrix = rotation_matrix_y(bbox3d[3])

    l, w, h = bbox3d[4:7]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(rot_matrix, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)
