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
from flask import Flask, Response, render_template
from pathlib import Path
import pandas as pd
from opendr.perception.object_detection_3d.datasets.kitti import KittiDataset, LabeledPointCloudsDatasetIterator

# OpenDR imports
from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import VoxelObjectDetection3DLearner
from opendr.engine.data import PointCloud

from data_generators import lidar_point_cloud_generator, disk_point_cloud_generator
from point_clouds import draw_point_cloud_bev

TEXT_COLOR = (255, 0, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
lock = threading.Lock()
lidar_data_generator = None


# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def runnig_fps(alpha=0.1):
    t0 = time.time_ns()
    fps_avg = 10

    def wrapped():
        nonlocal t0, alpha, fps_avg
        t1 = time.time_ns()
        delta = (t1 - t0) * 1e-9
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


def voxel_object_detection_3d(config_path, model_name=None):
    global lidar_data_generator, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    # Init model
    learner = VoxelObjectDetection3DLearner(config_path)

    if model_name is not None:
        learner.download(model_name, "./models")
    learner.load("./models/" + model_name, verbose=True)

    # dataset = KittiDataset("/data/sets/kitti_second")

    # dataset_path = "/data/sets/kitti_second"

    # val_dataset = LabeledPointCloudsDatasetIterator(
    #     dataset_path + "/training/velodyne",
    #     dataset_path + "/training/label_2",
    #     dataset_path + "/training/calib",
    # )

    # r = learner.eval(val_dataset)

    print("Learner created")

    # Loop over frames from the video stream
    while True:
        try:
            point_cloud: PointCloud = next(lidar_data_generator)
            print("Point cloud created")

            predictions = learner.infer(point_cloud)

            print("found", len(predictions), "objects")

            frame = draw_point_cloud_bev(point_cloud.data, predictions)
            draw_fps(frame, fps())

            print("frame created")

            with lock:
                output_frame = frame.copy()
        except Exception as e:
            print(e)


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
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
        "-s",
        "--source",
        type=str,
        default="disk",
        help="Data source",
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
    args = vars(ap.parse_args())

    lidar_data_generator = {
        "disk": disk_point_cloud_generator(args["data_path"])
    }[args["source"]]
    # time.sleep(2.0)

    algorithm = {
        "voxel": voxel_object_detection_3d,
    }[args["algorithm"]]

    # start a thread that will perform detection
    t = threading.Thread(target=algorithm, args=(args["model_config"], args["model_name"]))
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
