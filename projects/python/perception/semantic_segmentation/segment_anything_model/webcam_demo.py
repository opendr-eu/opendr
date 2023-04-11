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
import argparse
import time
import matplotlib.pyplot as plt
from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import SamLearner
from opendr.perception.object_detection_2d.yolov5.yolov5_learner import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.target import BoundingBox, BoundingBoxList


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    args = parser.parse_args()

    yolo_detector = YOLOv5DetectorLearner(model_name="yolov5l6.pt", device="cpu")

    sam = SamLearner(device=args.device)
    # sam.download(path=".", verbose=True)
    # sam.load("openpose_default")

    # Use the first camera available on the system
    image_provider = VideoReader(0)

    counter, avg_fps = 0, 0
    plt.ion()
    for img in image_provider:

        start_time = time.perf_counter()
        odr_img = Image(img)
        detections = yolo_detector.infer(odr_img)
        draw_bounding_boxes(img, detections, yolo_detector.classes, line_thickness=3)

        blended_img = None
        for d in detections:
            if yolo_detector.classes[int(d.name)] == "bottle":

                # Perform segmentation
                mask, scores, logits, bbox_prompt = sam.infer(img, d)

                blended_img = sam.draw(img, bbox_prompt, mask)

        end_time = time.perf_counter()
        fps = 1.0 / (end_time - start_time)
        # bbox = BoundingBox(left=bbox_prompt[0], top=bbox_prompt[1],
        #                    width=bbox_prompt[2] - bbox_prompt[0],
        #                    height=bbox_prompt[3] - bbox_prompt[1],
        #                    name=0,
        #                    score=1.0)
        # bbox_prompts = BoundingBoxList([])
        # bbox_prompts.add_box(bbox)

        # draw_bounding_boxes(img, bbox_prompts)

        if blended_img is not None:
            img_show = blended_img
        else:
            img_show = img
        # Calculate a running average on FPS
        avg_fps = 0.8 * fps + 0.2 * fps

        # Wait a few frames for FPS to stabilize
        if counter > 5:
            image = cv2.putText(img_show, "FPS: %.2f" % (avg_fps,), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Result', img_show)
        cv2.waitKey(1)
        counter += 1
