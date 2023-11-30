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

from opendr.perception.semantic_segmentation import YOLOv8SegLearner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--model", help="Model to use", type=str, default="yolov8s-seg",
                        choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg", "custom"])
    parser.add_argument("--model_path", help="Path to custom model if 'model' is set to 'custom'.", type=str,
                        default=None)
    args = parser.parse_args()

    yolov8_seg_learner = YOLOv8SegLearner(model_name=args.model, model_path=args.model_path, device=args.device)
    print(f"Using '{args.model}' model on {args.device}.")
    # Run inference
    # Add classes=["class_name", ...] argument to filter classes
    # Use print(yolov8_seg_learner.get_classes()) to see available class names
    # Providing a string can take advantage of the YOLOv8 built-in features
    # https://docs.ultralytics.com/modes/predict/#inference-sources
    heatmap = yolov8_seg_learner.infer("https://ultralytics.com/images/bus.jpg", no_mismatch=True, verbose=True)

    # Use yolov8 visualization
    visualization_img = yolov8_seg_learner.get_visualization()

    # Display the annotated frame
    cv2.imshow('Heatmap', visualization_img)
    print("Press any key to close OpenCV window...")
    cv2.waitKey(0)
