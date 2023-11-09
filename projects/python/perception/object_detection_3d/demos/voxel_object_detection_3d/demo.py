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
import os
import argparse
import threading
import time
import numpy as np
import cv2
from flask import Flask, Response, render_template, request

# OpenDR imports
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner
from opendr.perception.object_tracking_3d import ObjectTracking3DAb3dmotLearner
from data_generators import (
    lidar_point_cloud_generator,
    disk_point_cloud_generator,
)
from draw_point_clouds import (
    draw_point_cloud_bev,
    draw_point_cloud_projected_numpy,
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


def rplidar(*args, **kwargs):
    from rplidar_processor import RPLidar

    return RPLidar(*args, **kwargs)


def o3mlidar(*args, **kwargs):
    from o3m_lidar.o3m_lidar import O3MLidar

    return O3MLidar(*args, **kwargs)


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


def draw_dict(frame, dict, scale=5):

    i = 0

    for k, v in dict.items():
        cv2.putText(
            frame,
            f"{k}: {v}",
            (10, frame.shape[0] - 10 - 30 * scale * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            TEXT_COLOR,
            scale,
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

    with lock:
        output_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_dict(output_frame, {"Loading": "model"}, 1)

    # Prep stats
    fps = runnig_fps()

    predict = model_name is not None and model_name != "None"

    if predict:

        # Init model
        detection_learner = VoxelObjectDetection3DLearner(config_path)
        tracking_learner = ObjectTracking3DAb3dmotLearner()

        if model_name is not None and not os.path.exists(
            "./models/" + model_name
        ):
            detection_learner.download(model_name, "./models")
        detection_learner.load("./models/" + model_name, verbose=True)
        print("Learner created")

    else:
        detection_learner = None
        tracking_learner = None

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

    if lidar_type == "velodyne":
        xs = [-20, 90]
        ys = [-50, 50]
        scale = 20
        image_size_x = 1000
        image_size_y = 1000
        font_scale = 4
        tvec = np.array([10.8, 8.34, 16.8], dtype=np.float32)
        rvec = np.array([-10.67, 26.69, 6.914], dtype=np.float32)
        fx = 864.98
        fy = 864.98
    elif lidar_type == "rplidar":
        xs = [-10, 10]
        ys = [-10, 10]
        scale = 30
        image_size_x = 60
        image_size_y = 6

        tvec = np.array([10.8, 8.34, 16.8], dtype=np.float32)
        rvec = np.array([-10.67, 26.69, 6.914], dtype=np.float32)
        fx = 864.98
        fy = 864.98
        font_scale = 0.5
    elif lidar_type == "o3mlidar":
        xs = [-8, 8]
        ys = [-8, 8]
        scale = 40
        image_size_x = 600
        image_size_y = 600
        font_scale = 1
        tvec = np.array([4.8, 2.4, 13.2], dtype=np.float32)
        rvec = np.array([-6.28, 15.39, 5.03], dtype=np.float32)
        fx = 864.98
        fy = 864.98
    else:
        xs = [-20, 90]
        ys = [-50, 50]
        scale = 20
        image_size_x = 1000
        image_size_y = 3000
        font_scale = 4

    while True:
        try:

            t = time.time()

            point_cloud = next(point_cloud_generator)

            pc_time = time.time() - t

            if len(point_cloud.data) <= 0:
                continue

            t = time.time()

            if predict:
                predictions = detection_learner.infer(point_cloud)
                tracking_predictions = tracking_learner.infer(predictions)
            else:
                predictions = []
                tracking_predictions = []

            if len(predictions) > 0:
                print(
                    "found", len(predictions), "objects",
                )

            predict_time = time.time() - t
            t = time.time()

            frame_bev_2 = draw_point_cloud_bev(
                point_cloud.data, tracking_predictions, scale, xs, ys
            )
            frame_proj_2 = draw_point_cloud_projected_numpy(
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
            frame = stack_images([frame, frame_bev_2], "horizontal")

            draw_time = time.time() - t

            total_time = pc_time + predict_time + draw_time

            draw_dict(
                frame,
                {
                    "FPS": fps(),
                    "predict": str(int(predict_time * 100 / total_time)) + "%",
                    "get data": str(int(pc_time * 100 / total_time)) + "%",
                    "draw": str(int(draw_time * 100 / total_time)) + "%",
                },
                font_scale,
            )

            for key in keys_pressed:
                process_key(key)

            keys_pressed.clear()

            with lock:
                output_frame = frame.copy()
        except FileExistsError as e:
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
            b"Content-Type: image/jpeg\r\n\r\n" +
            bytearray(encodedImage) +
            b"\r\n"
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
        default="None",
        help="Pretrained model name",
    )
    ap.add_argument(
        "-mc",
        "--model_config",
        type=str,
        default="None",
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
    ap.add_argument(
        "-o3mp",
        "--o3m_port",
        type=int,
        default=42000,
        help="Port for O3M Lidar",
    )
    ap.add_argument(
        "-o3mip",
        "--o3m_ip",
        type=str,
        default="0.0.0.0",
        help="IP for O3M Lidar",
    )
    ap.add_argument(
        "-o3mbs",
        "--o3m_buffer_size",
        type=int,
        default=1460,
        help="Buffer size for O3M Lidar",
    )
    args = vars(ap.parse_args())

    point_cloud_generator = {
        "disk": lambda: disk_point_cloud_generator(
            args["data_path"], count=None
        ),
        "rplidar": lambda: lidar_point_cloud_generator(
            rplidar(args["rplidar_port"])
        ),
        "o3mlidar": lambda: lidar_point_cloud_generator(
            o3mlidar(
                ip=args["o3m_ip"],
                port=args["o3m_port"],
                buffer_size=args["o3m_buffer_size"],
            )
        ),
    }[args["source"]]()

    lidar_type = {
        "disk": "velodyne",
        "velodyne": "velodyne",
        "rplidar": "rplidar",
        "o3mlidar": "o3mlidar",
    }[args["source"]]

    algorithm = {"voxel": voxel_object_detection_3d}[args["algorithm"]]

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
