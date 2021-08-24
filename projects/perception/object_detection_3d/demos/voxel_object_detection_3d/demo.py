# Copyright 2020-2021 OpenDR European Project
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
import os
import argparse
import threading
import time
from typing import Dict
import numpy as np
import torch
import torchvision
import cv2
from imutils import resize
from flask import Flask, Response, render_template, request
from pathlib import Path
import pandas as pd

# from opendr.perception.object_detection_3d.datasets.kitti import KittiDataset, LabeledPointCloudsDatasetIterator

# OpenDR imports
from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner,
)
from opendr.perception.object_tracking_3d.ab3dmot.object_tracking_3d_ab3dmot_learner import (
    ObjectTracking3DAb3dmotLearner,
)
from opendr.engine.data import PointCloud

from data_generators import (
    lidar_point_cloud_generator,
    disk_point_cloud_generator,
)
from draw_point_clouds import (
    draw_point_cloud_bev,
    draw_point_cloud_projected,
    draw_point_cloud_projected_2,
)

TEXT_COLOR = (255, 112, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
lock = threading.Lock()
point_cloud_generator = None
keys_pressed = []

lidar_type = "velodyne"


# initialize a flask object
app = Flask(__name__)


def rplidar(*args):
    from rplidar_processor import RPLidar

    return RPLidar(*args)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def runnig_fps(alpha=0.1):
    t0 = time.time()
    fps_avg = 10

    def wrapped():
        nonlocal t0, alpha, fps_avg
        t1 = time.time()
        delta = t1 - t0
        t0 = t1
        fps_avg = alpha * (1 / delta) + (1 - alpha) * fps_avg
        return fps_avg

    return wrapped


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        1,
    )


def draw_dict(frame, dict):

    i = 0

    for k, v in dict.items():
        cv2.putText(
            frame,
            f"{k}: {v}",
            (10, frame.shape[0] - 10 - 30 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOR,
            1,
        )
        i += 1


def stack_images(images, mode="horizontal"):

    max_width, max_height = 0, 0

    for image in images:
        width, height, _ = image.shape
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    if mode == "horizontal":
        for i in range(len(images)):
            width, _, _ = images[i].shape

            delta = max_width - width
            pad = delta // 2

            images[i] = np.pad(
                images[i], [(pad, pad + delta % 2), (0, 0), (0, 0)]
            )

        return cv2.hconcat(images)
    elif mode == "vertical":
        for i in range(len(images)):
            _, height, _ = images[i].shape

            delta = max_height - height
            pad = delta // 2

            images[i] = np.pad(
                images[i], [(0, 0), (pad, pad + delta % 2), (0, 0)]
            )

        return cv2.vconcat(images)


def voxel_object_detection_3d(config_path, model_name=None):
    global point_cloud_generator, output_frame, lock, lidar_type

    # Prep stats
    fps = runnig_fps()

    # Init model
    detection_learner = VoxelObjectDetection3DLearner(config_path)

    if model_name is not None and not os.path.exists(
        "./models/" + model_name
    ):
        detection_learner.download(model_name, "./models")
    detection_learner.load("./models/" + model_name, verbose=True)

    tracking_learner = ObjectTracking3DAb3dmotLearner()

    # dataset = KittiDataset("/data/sets/kitti_second")

    # dataset_path = "/data/sets/kitti_second"

    # val_dataset = LabeledPointCloudsDatasetIterator(
    #     dataset_path + "/training/velodyne",
    #     dataset_path + "/training/label_2",
    #     dataset_path + "/training/calib",
    # )

    # r = learner.eval(val_dataset)

    # tvec = np.array([0, 0, 0], dtype=np.float32)
    # rvec = np.array([0, 0, 0], dtype=np.float32)
    # fx = 10
    # fy = 10
    # tvec = np.array([-1.25, 4.71, -12], dtype=np.float32)
    # rvec = np.array([2.4, 15.6, 10.8], dtype=np.float32)
    # fx = 864
    # fy = 384
    tvec0 = np.array([0, 4.8, 2.4], dtype=np.float32)
    tvec = np.array([2.4, 22.8, 13.20], dtype=np.float32)
    rvec0 = np.array([-5.33, 15.39, 6.6], dtype=np.float32)
    rvec = np.array([-6.28, 15.39, 5.03], dtype=np.float32)
    fx = 864.98
    fy = 384.43
    # tvec = np.array([-10.8, -16.8, -12], dtype=np.float32)
    # rvec = np.array([-2.32, 0.6, -1.2], dtype=np.float32)
    # fx = 384
    # fy = 384

    def process_key(key):

        nonlocal tvec, rvec, fx, fy

        dt = 1.2
        dr = math.pi / 10

        if key == 2:
            tvec += np.array([0.00, dt, 0.00], dtype=np.float32)
        elif key == 3:
            tvec += np.array([-dt, 0.00, 0.00], dtype=np.float32)
        elif key == 0:
            tvec += np.array([0.00, -dt, 0.00], dtype=np.float32)
        elif key == 1:
            tvec += np.array([dt, 0.00, 0.00], dtype=np.float32)

        if key == 4:
            rvec += np.array([0.00, dr, 0.00], dtype=np.float32)
        elif key == 5:
            rvec += np.array([-dr, 0.00, 0.00], dtype=np.float32)
        elif key == 6:
            rvec += np.array([0.00, -dr, 0.00], dtype=np.float32)
        elif key == 7:
            rvec += np.array([dr, 0.00, 0.00], dtype=np.float32)
        elif key == 8:
            rvec += np.array([0.00, 0.00, -dr], dtype=np.float32)
        elif key == 9:
            rvec += np.array([0.00, 0.00, dr], dtype=np.float32)

        elif key == 10:
            fx /= 1.5
        elif key == 11:
            fx *= 1.5
        elif key == 12:
            fy /= 1.5
        elif key == 13:
            fy *= 1.5

        elif key == 14:
            tvec += np.array([0.00, 0.00, dt], dtype=np.float32)
        elif key == 15:
            tvec += np.array([0.00, 0.00, -dt], dtype=np.float32)

        elif key == 98:
            tvec = np.array([0.00, 0.00, 0.00], dtype=np.float32)
        elif key == 99:
            rvec = np.array([0.00, 0.00, 0.00], dtype=np.float32)
        elif key == 100:
            tvec = np.array([0.00, 0.00, 0.00], dtype=np.float32)
            rvec = np.array([0.00, 0.00, 0.00], dtype=np.float32)
            fx = 10
            fy = 10

    print("Learner created")

    if lidar_type == "velodyne":
        xs = [-20, 90]
        ys = [-50, 50]
        scale = 20
        # image_size_x = 600
        # image_size_y = 1800
        image_size_x = 1000
        image_size_y = 3000
    elif lidar_type == "rplidar":
        xs = [-10, 10]
        ys = [-10, 10]
        scale = 30
        image_size_x = 60
        image_size_y = 60
    else:
        xs = [-90, 90]
        ys = [-90, 90]
        scale = 10
        image_size_x = 600
        image_size_y = 600

    # Loop over frames from the video stream
    while True:
        try:
            point_cloud: PointCloud = next(point_cloud_generator)
            # print("Point cloud created")

            predictions = detection_learner.infer(point_cloud)
            tracking_predictions = tracking_learner.infer(predictions)
            # predictions = []

            print("found", len(predictions), "objects", "and", len(tracking_predictions), "tracklets")

            frame_bev = draw_point_cloud_bev(
                point_cloud.data, tracking_predictions, scale, xs, ys
            )
            frame_bev_2 = draw_point_cloud_bev(
                point_cloud.data, predictions, scale, xs, ys
            )

            # frame_proj = draw_point_cloud_projected_2(
            #     point_cloud.data,
            #     predictions,
            #     tvec=tvec0,
            #     rvec=rvec0,
            #     image_size_x=image_size_x,
            #     image_size_y=image_size_y,
            #     fx=fx,
            #     fy=fy,
            # )
            frame_proj_2 = draw_point_cloud_projected_2(
                point_cloud.data,
                predictions,
                tvec=tvec,
                rvec=rvec,
                image_size_x=image_size_x,
                image_size_y=image_size_y,
                fx=fx,
                fy=fy,
            )
            frame = frame_proj_2
            # frame = stack_images([frame_proj, frame], "vertical")
            # frame = stack_images([frame, frame_bev], "vertical")
            frame = stack_images([frame, frame_bev], "horizontal")
            frame = stack_images([frame, frame_bev_2], "horizontal")

            draw_dict(
                frame,
                {"FPS": fps(), "tvec": tvec, "rvec": rvec, "f": [fx, fy],},
            )

            # print("frame created")

            for key in keys_pressed:
                process_key(key)

            keys_pressed.clear()

            with lock:
                output_frame = frame.copy()
        except Exception as e:
            print(e)
            # torch.cuda.empty_cache()
            raise e


def generate():
    # grab global references to the output frame and lock variables
    global output_frame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if output_frame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + bytearray(encodedImage)
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/keypress", methods=["POST"])
def process_keypress():

    global keys_pressed

    data = request.get_json()
    key = data["key"]
    keys_pressed.append(key)

    return ("", 204)


# check to see if this is the main thread of execution
if __name__ == "__main__":
    # construct the argument parser and parse command line arguments

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--ip", type=str, required=True, help="IP address of the device"
    )
    ap.add_argument(
        "-o",
        "--port",
        type=int,
        required=True,
        help="Ephemeral port number of the server (1024 to 65535)",
    )
    ap.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="pointpillars_ped_cycle_xyres_16",
        help="Pretrained model name",
    )
    ap.add_argument(
        "-mc",
        "--model_config",
        type=str,
        default=os.path.join(
            "configs", "pointpillars_ped_cycle_xyres_16.proto"
        ),
        help="Model configuration file",
    )
    ap.add_argument(
        "-s", "--source", type=str, default="disk", help="Data source",
    )
    ap.add_argument(
        "-dp",
        "--data_path",
        type=str,
        default="",
        help="Path for disk-based data generators",
    )
    ap.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="voxel",
        help="Which algortihm to run",
        choices=["voxel"],
    )
    ap.add_argument(
        "-rpp",
        "--rplidar_port",
        type=str,
        default="",
        help="Port for RPLidar",
    )
    args = vars(ap.parse_args())

    point_cloud_generator = {
        "disk": lambda: disk_point_cloud_generator(
            args["data_path"], count=None
        ),
        "rplidar": lambda: lidar_point_cloud_generator(
            rplidar(args["rplidar_port"])
        ),
    }[args["source"]]()
    # time.sleep(2.0)

    lidar_type = {
        "disk": "velodyne",
        "velodyne": "velodyne",
        "rplidar": "rplidar",
    }[args["source"]]

    algorithm = {"voxel": voxel_object_detection_3d,}[args["algorithm"]]

    # start a thread that will perform detection
    t = threading.Thread(
        target=algorithm, args=(args["model_config"], args["model_name"])
    )
    t.daemon = True
    t.start()

    # start the flask app
    app.run(
        host=args["ip"],
        port=args["port"],
        debug=True,
        threaded=True,
        use_reloader=False,
    )
