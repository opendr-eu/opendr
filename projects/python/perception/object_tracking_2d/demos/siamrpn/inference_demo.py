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

import cv2
from opendr.engine.target import TrackingAnnotation
from opendr.perception.object_tracking_2d import SiamRPNLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    learner = SiamRPNLearner(device=args.device)
    learner.download(".", mode="pretrained")
    learner.load("siamrpn_opendr")

    learner.download(".", mode="video")
    cap = cv2.VideoCapture("tc_Skiing_ce.mp4")

    init_bbox = TrackingAnnotation(left=598, top=312, width=75, height=200, name=0, id=0)

    frame_no = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no == 0:
            # first frame, pass init_bbox to infer function to initialize the tracker
            pred_bbox = learner.infer(frame, init_bbox)
        else:
            # after the first frame only pass the image to infer
            pred_bbox = learner.infer(frame)

        frame_no += 1

        cv2.rectangle(frame, (pred_bbox.left, pred_bbox.top),
                      (pred_bbox.left + pred_bbox.width, pred_bbox.top + pred_bbox.height),
                      (0, 255, 255), 3)
        cv2.imshow('Tracking Result', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
