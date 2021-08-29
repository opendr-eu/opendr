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

import argparse
import os
from data_generators import camera_image_generator, disk_image_generator
import threading
import time
from typing import Dict
import numpy as np
import torch
import torchvision
import cv2
from imutils import resize
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from pathlib import Path
import pandas as pd
from opendr.engine.target import TrackingAnnotation, TrackingAnnotation3D, TrackingAnnotation3DList, TrackingAnnotationList

# OpenDR imports
from opendr.perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import ObjectTracking2DFairMotLearner
from opendr.engine.data import Video, Image

TEXT_COLOR = (255, 0, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
image_generator = None
lock = threading.Lock()
colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]


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


def draw_predictions(frame, predictions: TrackingAnnotationList):
    global colors


    for prediction in predictions.boxes:
        prediction: TrackingAnnotation = prediction

        color = colors[prediction.id * 10 % len(colors)]

        cv2.rectangle(frame, (int(prediction.top), int(prediction.left)), (int(prediction.top + prediction.width), int(prediction.left + prediction.height)), color, 2)

def video_har_preprocessing(image_size: int, window_size: int):
    frames = []

    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal frames, standardize
        frame = resize(frame, height=image_size, width=image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        if not frames:
            frames = [frame for _ in range(window_size)]
        else:
            frames.pop(0)
            frames.append(frame)
        vid = Video(torch.stack(frames, dim=1))
        return vid

    return wrapped


def fair_mot_tracking(model_name, device):
    global vs, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    predict = model_name is not None and model_name != "None"

    if predict:

        # Init model
        learner = ObjectTracking2DFairMotLearner(device=device)
        if not os.path.exists(
            "./models/" + model_name
        ):
            learner.download(model_name, "./models")
        learner.load("./models/" + model_name, verbose=True)

        print("Learner created")
    else:
        learner = None

    # Loop over frames from the video stream
    while True:
        try:
            image: Image = next(image_generator)

            if predict:
                predictions = learner.infer(image)
                print("Found", len(predictions), "objects")

            frame = np.moveaxis(image.data, [0, 1, 2], [2, 0, 1]).copy()
            
            if predict:
                draw_predictions(frame, predictions)

            frame = cv2.flip(frame, 1)
            draw_fps(frame, fps())

            with lock:
                output_frame = frame.copy()
        except Exception:
            pass


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
        choices=["fair_mot"],
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
        "camera": lambda: camera_image_generator(
            VideoStream(src=args["video_source"]).start()
        ),
    }[args["source"]]()

    time.sleep(2.0)

    algorithm = {
        "fair_mot": fair_mot_tracking,
    }[args["algorithm"]]

    # start a thread that will perform motion detection
    t = threading.Thread(target=algorithm, args=(args["model_name"], args["device"]))
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

