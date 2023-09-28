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
import time

import cv2

from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import YOLOv8SegLearner


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
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--model", help="Model to use", type=str, default="yolov8s-seg",
                        choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"])
    args = parser.parse_args()

    yolov8_seg_learner = YOLOv8SegLearner(model_name=args.model)

    # Use the first camera available on the system
    image_provider = VideoReader(0)
    fps = -1.0
    # try:
    counter, avg_fps = 0, 0
    for img in image_provider:

        img = Image(img)

        start_time = time.perf_counter()

        # Run inference
        # Add classes=["class_name", ...] argument to filter classes
        # Use print(yolov8_seg_learner.get_classes()) to see available class names
        heatmap = yolov8_seg_learner.infer(img, no_mismatch=True)

        end_time = time.perf_counter()
        fps = 1.0 / (end_time - start_time)
        # Calculate a running average on FPS
        avg_fps = 0.8 * fps + 0.2 * avg_fps

        annotated_frame = yolov8_seg_learner.get_visualization()
        # Alternative visualization:
        # import cv2
        # from matplotlib import cm
        # # Create a color map and translate colors
        # segmentation_mask = heatmap.data
        #
        # colormap = cm.get_cmap('viridis', 81).colors
        # segmentation_img = np.uint8(255 * colormap[segmentation_mask][:, :, :3])
        #
        # # Blend original image and the segmentation mask
        # blended_img = np.uint8(0.4 * img.opencv() + 0.6 * segmentation_img)
        #
        # cv2.imshow('Heatmap', blended_img)
        # cv2.waitKey(1)

        # Wait a few frames for FPS to stabilize
        if counter < 5:
            counter += 1
        else:
            annotated_frame = cv2.putText(annotated_frame, "FPS: %.2f" % (avg_fps,), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey(1)
    # except:
    #     print("Inference fps: ", round(fps))
