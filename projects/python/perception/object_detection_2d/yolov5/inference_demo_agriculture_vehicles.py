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
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Model name or path", type=str, default='yolov5s_finetuned_in_trucks.pt')
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Initialize the YOLOv5 detector with the given model and device
    yolo = YOLOv5DetectorLearner(model_name=args.model_name, device=args.device, path="./"+args.model_name)
    yolo.download(".", mode="images", verbose=True, img_name="truck4.jpg")
    yolo.download(".", mode="images", verbose=True, img_name="truck7.jpg")

    im1 = Image.open('truck4.jpg')
    im2 = Image.open('truck7.jpg')

    results = yolo.infer(im1)
    draw_bounding_boxes(im1.opencv(), results, yolo.classes, show=True, line_thickness=3)

    results = yolo.infer(im2)
    draw_bounding_boxes(im2, results, yolo.classes, show=True, line_thickness=3)