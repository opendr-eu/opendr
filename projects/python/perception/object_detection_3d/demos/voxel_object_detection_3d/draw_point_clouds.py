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

import cv2
import numpy as np
import math
from metrics import MinMetric, MaxMetric
from PIL import Image, ImageDraw, ImageFont

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core.box_np_ops import (
    center_to_corner_box3d,
)

box_draw_indicies = [
    (1, 4),
    (2, 5),
    (3, 6),
    (0, 7),
    (5,),
    (6,),
    (7,),
    (4,),
]


def draw_point_cloud_bev(
    point_cloud, predictions=None, scale=10, xs=[-90, 90], ys=[-90, 90]
):
    x_min = MinMetric()
    y_min = MinMetric()
    x_max = MaxMetric()
    y_max = MaxMetric()

    x_min.update(xs[0])
    x_max.update(xs[1])
    y_min.update(ys[0])
    y_max.update(ys[1])

    point_cloud = point_cloud[point_cloud[:, 0] > x_min.get()]
    point_cloud = point_cloud[point_cloud[:, 0] < x_max.get()]
    point_cloud = point_cloud[point_cloud[:, 1] > y_min.get()]
    point_cloud = point_cloud[point_cloud[:, 1] < y_max.get()]

    x_size = x_max.get() - x_min.get()
    y_size = y_max.get() - y_min.get()

    image_size_x = int(x_size * scale + 1)
    image_size_y = int(y_size * scale + 1)

    point_cloud_x = (
        image_size_x - 1 - (point_cloud[:, 0] - x_min.get()) * scale
    ).astype(np.int32)
    point_cloud_y = (
        image_size_y - 1 - (point_cloud[:, 1] - y_min.get()) * scale
    ).astype(np.int32)

    colors = np.array([255, 255, 255], dtype=np.uint8)

    color_image = np.zeros([image_size_x, image_size_y, 3], dtype=np.uint8)

    def rotate_rectangle(x, y, half_size_x, half_size_y, angle):
        def distance(ax, ay, bx, by):
            return math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)

        # rotates point `A` about point `B` by `angle` radians clockwise.
        def rotated_about(ax, ay, bx, by, angle):
            radius = distance(ax, ay, bx, by)
            angle += math.atan2(ay - by, ax - bx)
            return (
                round(bx + radius * math.cos(angle)),
                round(by + radius * math.sin(angle)),
            )

        vertices = (
            (x + half_size_x, y + half_size_y),
            (x + half_size_x, y - half_size_y),
            (x - half_size_x, y - half_size_y),
            (x - half_size_x, y + half_size_y),
        )

        result = [rotated_about(vx, vy, x, y, angle) for vx, vy in vertices]

        return [(y, x) for (x, y) in result]

    pil_image = Image.new(
        "RGB", (image_size_y + 1, image_size_x + 1), color="black"
    )
    pil_draw = ImageDraw.Draw(pil_image)

    font = None

    for box in predictions.boxes:

        x_3d, y_3d = box.location[:2]
        size = box.dimensions
        rotation = box.rotation_y
        id = box.id if hasattr(box, "id") else None
        x_bev = (image_size_x - 1 - (x_3d - x_min.get()) * scale).astype(
            np.int32
        )
        y_bev = (image_size_y - 1 - (y_3d - y_min.get()) * scale).astype(
            np.int32
        )

        half_size_x, half_size_y = (size[:2] * scale / 2).astype(np.int32)

        pil_draw.polygon(
            rotate_rectangle(x_bev, y_bev, half_size_x, half_size_y, rotation),
            fill=(192, 102, 50),
            outline=(255, 0, 255),
        )
        if id is not None:

            if font is None:
                font = ImageFont.truetype("./fonts/arial.ttf", 40)

            pil_draw.text((y_bev, x_bev), str(id), font=font, align="center")

    color_image = np.array(pil_image)
    color_image[point_cloud_x, point_cloud_y] = colors

    return color_image


def draw_point_cloud_projected_cv(
    point_cloud,
    predictions=None,
    image_size_x=600,
    image_size_y=600,
    rvec=np.array([0, 0, 0], dtype=np.float32),
    tvec=np.array([0, 0, 0], dtype=np.float32),
    fx=10,
    fy=10,
):

    cameraMatrix = np.array(
        [[fx, 0, 1], [0, fy, 1], [0, 0, 1]], dtype=np.float32
    )
    distCoef = None

    if len(point_cloud) > 0:

        pc = point_cloud[:, :3].astype(np.float64)
        pc = pc[:, [1, 0, 2]]

        projectionsAndJacobians = cv2.projectPoints(
            pc,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoef,
        )

        projections = (
            projectionsAndJacobians[0].reshape(-1, 2).astype(np.int32)
        )

        projections = projections[projections[:, 0] >= 0]
        projections = projections[projections[:, 1] >= 0]
        projections = projections[projections[:, 0] < image_size_x]
        projections = projections[projections[:, 1] < image_size_y]
    else:
        projections = np.zeros((0, 2), dtype=np.int32)

    colors = np.array([255, 255, 255], dtype=np.uint8)

    color_image = np.zeros([image_size_x, image_size_y, 3], dtype=np.uint8)
    color_image[projections[:, 0], projections[:, 1]] = colors

    return color_image


def draw_point_cloud_projected_numpy(
    point_cloud,
    predictions=None,
    image_size_x=600,
    image_size_y=600,
    rvec=np.array([0, 0, 0], dtype=np.float32),
    tvec=np.array([0, 0, 0], dtype=np.float32),
    fx=10,
    fy=10,
):

    cameraMatrix = np.array(
        [[fx, 0, 1], [0, fy, 1], [0, 0, 1]], dtype=np.float32
    )
    distCoef = None

    # R, t - current 6dof pose of the camera
    # K - 3x3 Camera matrix
    # D - distortion coefficients
    # xyzs - Nx3 3d points

    pc = point_cloud[:, :3].astype(np.float64)
    pc = pc[:, [1, 0, 2]]

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    K = cameraMatrix
    D = distCoef

    def project(xyzs, drop):
        nonlocal R, t, K, D

        proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
        # convert 3D points into homgenous points
        xyz_hom = np.hstack((xyzs, np.ones((xyzs.shape[0], 1))))

        xy_hom = np.dot(proj_mat, xyz_hom.T).T

        # get 2d coordinates in image [pixels]
        z = xy_hom[:, -1]
        xy = xy_hom[:, :2] / np.tile(z[:, np.newaxis], (1, 2))

        # drop all points behind camera
        if drop:
            xy = xy[z > 0]
        projections = xy.astype(np.int32)

        return projections

    if len(point_cloud) > 0:
        projections = project(pc, True)

        if len(predictions) > 0:
            prediction_locations = np.stack(
                [b.location for b in predictions.boxes]
            )
            prediction_dimensions = np.stack(
                [b.dimensions for b in predictions.boxes]
            )
            prediction_rotations = np.stack(
                [b.rotation_y for b in predictions.boxes]
            )

            prediction_corners = center_to_corner_box3d(
                prediction_locations,
                prediction_dimensions,
                prediction_rotations,
                [0.5, 0.5, 0],
                2,
            )
        else:
            prediction_corners = np.zeros((0, 8, 3), dtype=np.float32)
            prediction_locations = np.zeros((0, 3), dtype=np.float32)

        projections = projections[projections[:, 0] >= 0]
        projections = projections[projections[:, 1] >= 0]
        projections = projections[projections[:, 0] < image_size_x]
        projections = projections[projections[:, 1] < image_size_y]
    else:
        projections = np.zeros((0, 2), dtype=np.int32)
        prediction_corners = np.zeros((0, 8, 3), dtype=np.float32)
        prediction_locations = np.zeros((0, 3), dtype=np.float32)

    colors = np.array([255, 255, 255], dtype=np.uint8)

    color_image = np.zeros([image_size_x, image_size_y, 3], dtype=np.uint8)
    color_image[projections[:, 0], projections[:, 1]] = colors

    prediction_corners = prediction_corners[:, :, [1, 0, 2]]

    for corners in prediction_corners:
        projected_corners = project(corners, False)
        projected_corners = projected_corners[:, [1, 0]]

        for i, links in enumerate(box_draw_indicies):
            for p in links:
                cv2.line(
                    color_image,
                    tuple(projected_corners[i]),
                    tuple(projected_corners[p]),
                    (255, 0, 0),
                    4,
                )
    for l in project(prediction_locations[:, [1, 0, 2]], False):
        cv2.circle(color_image, tuple(l[[1, 0]]), 10, (0, 11, 125), 10)

    return color_image
