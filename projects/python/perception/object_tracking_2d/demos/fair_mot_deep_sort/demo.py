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

import argparse
import os
from data_generators import camera_image_generator, disk_image_generator, disk_image_with_detections_generator
import threading
import time
import numpy as np
import cv2
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from opendr.engine.target import (
    TrackingAnnotationList,
)

# OpenDR imports
from opendr.perception.object_tracking_2d import ObjectTracking2DFairMotLearner
from opendr.perception.object_tracking_2d import ObjectTracking2DDeepSortLearner

TEXT_COLOR = (255, 0, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
image_generator = None
lock = threading.Lock()
colors = [
    (255, 0, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (35, 69, 55),
    (43, 63, 54),
    (37, 70, 54),
    (50, 67, 54),
    (51, 66, 49),
    (43, 75, 64),
    (55, 65, 42),
    (53, 63, 42),
    (43, 46, 38),
    (41, 41, 36),
    (70, 54, 35),
    (70, 54, 41),
    (65, 54, 40),
    (63, 55, 38),
    (63, 54, 35),
    (83, 73, 49),
    (81, 65, 45),
    (75, 65, 42),
    (85, 74, 60),
    (79, 64, 55),
    (75, 67, 59),
    (74, 75, 70),
    (70, 71, 62),
    (57, 62, 46),
    (68, 54, 45),
    (66, 52, 43),
    (69, 54, 43),
    (73, 59, 47),
    (30, 52, 66),
    (41, 55, 65),
    (36, 54, 64),
    (44, 87, 120),
    (124, 129, 124),
    (109, 120, 118),
    (119, 132, 142),
    (105, 125, 137),
    (108, 94, 83),
    (93, 78, 70),
    (90, 76, 66),
    (90, 76, 66),
    (90, 77, 65),
    (91, 82, 68),
    (85, 77, 66),
    (84, 79, 58),
    (133, 113, 88),
    (130, 127, 121),
    (120, 109, 95),
    (112, 110, 102),
    (113, 110, 97),
    (103, 109, 99),
    (122, 124, 118),
    (198, 234, 221),
    (194, 230, 236),
]


# initialize a flask object
app = Flask(__name__)


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


def draw_predictions(frame, predictions: TrackingAnnotationList, is_centered=False, is_flipped_xy=True):
    global colors
    w, h, _ = frame.shape

    for prediction in predictions.boxes:
        prediction = prediction

        if not hasattr(prediction, "id"):
            prediction.id = 0

        color = colors[int(prediction.id) * 7 % len(colors)]

        x = prediction.left
        y = prediction.top

        if is_flipped_xy:
            x = prediction.top
            y = prediction.left

        if is_centered:
            x -= prediction.width
            y -= prediction.height

        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (
                int(x + prediction.width),
                int(y + prediction.height),
            ),
            color,
            2,
        )


def fair_mot_tracking(model_name, device):
    global vs, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    predict = model_name is not None and model_name != "None"

    if predict:

        # Init model
        learner = ObjectTracking2DFairMotLearner(device=device, use_pretrained_backbone=False)
        if not os.path.exists("./models/" + model_name):
            learner.download(model_name, "./models")
        learner.load("./models/" + model_name, verbose=True)

        print("Learner created")
    else:
        learner = None

    # Loop over frames from the video stream
    while True:
        try:

            t = time.time()
            image = next(image_generator)
            image_time = time.time() - t

            t = time.time()
            if predict:
                predictions = learner.infer(image)
                print("Found", len(predictions), "objects")
            predict_time = time.time() - t
            t = time.time()

            frame = np.ascontiguousarray(
                np.moveaxis(image.data, [0, 1, 2], [2, 0, 1]).copy()
            )

            if predict:
                draw_predictions(frame, predictions)

            frame = cv2.flip(frame, 1)
            draw_time = time.time() - t

            total_time = predict_time + image_time + draw_time

            draw_dict(
                frame,
                {
                    "FPS": fps(),
                    "predict": str(int(predict_time * 100 / total_time)) + "%",
                    "get data": str(int(image_time * 100 / total_time)) + "%",
                    "draw": str(int(draw_time * 100 / total_time)) + "%",
                    # "tvec": tvec, "rvec": rvec, "f": [fx, fy],
                },
                1,
            )

            with lock:
                output_frame = frame.copy()
        except Exception as e:
            print(e)
            raise e


def deep_sort_tracking(model_name, device):
    global vs, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    predict = model_name is not None and model_name != "None"

    if predict:
        learner = ObjectTracking2DDeepSortLearner(device=device)
        if not os.path.exists("./models/" + model_name):
            learner.download(model_name, "./models")
        learner.load("./models/" + model_name, verbose=True)

        print("Learner created")
    else:
        learner = None

    # Loop over frames from the video stream
    while True:
        try:

            t = time.time()
            image_with_detections = next(image_generator)
            image_time = time.time() - t

            t = time.time()
            if predict:
                predictions = learner.infer(image_with_detections)
                print("Found", len(predictions), "objects")
            predict_time = time.time() - t
            t = time.time()

            frame = np.ascontiguousarray(
                np.moveaxis(image_with_detections.data, [0, 1, 2], [2, 0, 1]).copy()
            )

            if predict:
                draw_predictions(frame, predictions, is_centered=False, is_flipped_xy=False)

            frame = cv2.flip(frame, 1)
            draw_time = time.time() - t

            total_time = predict_time + image_time + draw_time

            draw_dict(
                frame,
                {
                    "FPS": fps(),
                    "predict": str(int(predict_time * 100 / total_time)) + "%",
                    "get data": str(int(image_time * 100 / total_time)) + "%",
                    "draw": str(int(draw_time * 100 / total_time)) + "%",
                },
                1,
            )

            with lock:
                output_frame = frame.copy()
        except Exception as e:
            print(e)
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
        "-m",
        "--model_name",
        type=str,
        default="fairmot_dla34",
        help="Model identifier",
    )
    ap.add_argument(
        "-dp",
        "--data_path",
        type=str,
        default="",
        help="Path for disk-based data generators",
    )
    ap.add_argument(
        "-ds",
        "--data_splits",
        type=str,
        default="",
        help="Path for mot dataset splits",
    )
    ap.add_argument(
        "-s", "--source", type=str, default="disk", help="Data source",
    )
    ap.add_argument(
        "-v",
        "--video_source",
        type=int,
        default=0,
        help="ID of the video source to use",
    )
    ap.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="fair_mot",
        help="Which algortihm to run",
        choices=["fair_mot", "deep_sort"],
    )
    ap.add_argument(
        "-dev",
        "--device",
        type=str,
        default="cuda",
        help="Which device to use",
    )
    args = vars(ap.parse_args())

    image_generator = {
        "disk": lambda: disk_image_generator(
            args["data_path"], {"mot20": args["data_splits"]}, count=None
        ),
        "disk_with_detections": lambda: disk_image_with_detections_generator(
            args["data_path"], {"mot20": args["data_splits"]}, count=None
        ),
        "camera": lambda: camera_image_generator(
            VideoStream(src=args["video_source"]).start()
        ),
    }[args["source"]]()

    time.sleep(2.0)

    algorithm = {
        "fair_mot": fair_mot_tracking,
        "deep_sort": deep_sort_tracking,
    }[args["algorithm"]]

    # start a thread that will perform motion detection
    t = threading.Thread(
        target=algorithm, args=(args["model_name"], args["device"])
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
