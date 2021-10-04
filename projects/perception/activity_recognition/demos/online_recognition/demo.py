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

# OpenDR imports
from opendr.perception.activity_recognition.x3d.x3d_learner import X3DLearner
from opendr.perception.activity_recognition.cox3d.cox3d_learner import CoX3DLearner
from opendr.engine.data import Video, Image

TEXT_COLOR = (255, 0, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
lock = threading.Lock()


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


def draw_preds(frame, preds: Dict):
    base_skip = 40
    delta_skip = 30
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(
            frame,
            f"{prob:04.3f} {cls}",
            (10, base_skip + i * delta_skip),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOR,
            1,
        )


def draw_centered_box(frame, border):
    border = 10
    minX = (frame.shape[1] - frame.shape[0]) // 2 + border
    minY = border
    maxX = (frame.shape[1] + frame.shape[0]) // 2 - border
    maxY = frame.shape[0] - border
    cv2.rectangle(frame, (minX, minY), (maxX, maxY), color=TEXT_COLOR, thickness=1)


def center_crop(frame):
    height, width = frame.shape[0], frame.shape[1]
    e = min(height, width)
    x0 = (width - e) // 2
    y0 = (height - e) // 2
    cropped_frame = frame[y0:y0 + e, x0:x0 + e]
    return cropped_frame


def image_har_preprocessing(image_size: int):
    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal standardize
        frame = resize(frame, height=image_size, width=image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        return Image(frame, dtype=np.float)

    return wrapped


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


KINETICS400_CLASSES = pd.read_csv(
    "activity_recognition/kinetics400_classes.csv", verbose=True, index_col=0
).to_dict()["name"]


def clean_kinetics_preds(preds):
    k = 3
    class_scores, class_inds = torch.topk(preds[0].confidence, k=k)
    preds = {
        KINETICS400_CLASSES[int(class_inds[i])]: float(class_scores[i].item())
        for i in range(k)
    }
    return preds


def x3d_activity_recognition(model_name):
    global vs, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    # Init model
    learner = X3DLearner(device="cpu", backbone=model_name, num_workers=0)
    X3DLearner.download(path="model_weights", model_names={model_name})
    learner.load(Path("model_weights") / f"x3d_{model_name}.pyth")

    preprocess = video_har_preprocessing(
        image_size=learner.model_hparams["image_size"],
        window_size=learner.model_hparams["frames_per_clip"],
    )

    # Loop over frames from the video stream
    while True:
        try:
            frame = vs.read()

            frame = center_crop(frame)

            # Prepocess frame
            vid = preprocess(frame)

            # Gererate preds
            preds = learner.infer(vid)
            preds = clean_kinetics_preds(preds)

            frame = cv2.flip(frame, 1)  # Flip horizontally for webcam-compatibility
            draw_preds(frame, preds)
            draw_fps(frame, fps())

            with lock:
                output_frame = frame.copy()
        except Exception:
            pass


def cox3d_activity_recognition(model_name):
    global vs, output_frame, lock

    # Prep stats
    fps = runnig_fps()

    # Init model
    learner = CoX3DLearner(device="cpu", backbone=model_name, num_workers=0)
    CoX3DLearner.download(path="model_weights", model_names={model_name})
    learner.load(Path("model_weights") / f"x3d_{model_name}.pyth")

    preprocess = image_har_preprocessing(image_size=learner.model_hparams["image_size"])

    # Loop over frames from the video stream
    while True:
        try:
            frame = vs.read()

            frame = center_crop(frame)

            # Prepocess frame
            vid = preprocess(frame)

            # Gererate preds
            preds = learner.infer(vid)
            preds = clean_kinetics_preds(preds)

            frame = cv2.flip(frame, 1)  # Flip horizontally for webcam-compatibility
            draw_preds(frame, preds)
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
        default="xs",
        help="Model identifier",
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
        default="x3d",
        help="Which algortihm to run",
        choices=["cox3d", "x3d"],
    )
    args = vars(ap.parse_args())

    # initialize video stream and allow the camera sensor to warmup
    # vs = VideoStream(usePiCamera=1).start()
    vs = VideoStream(src=args["video_source"]).start()
    time.sleep(2.0)

    algorithm = {
        "x3d": x3d_activity_recognition,
        "cox3d": cox3d_activity_recognition,
    }[args["algorithm"]]

    # start a thread that will perform motion detection
    t = threading.Thread(target=algorithm, args=(args["model_name"],))
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

    # release the video stream pointer
    vs.stop()
