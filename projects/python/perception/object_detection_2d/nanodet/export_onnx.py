# Copyright 2020-2022 OpenDR European Project
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
from opendr.perception.object_detection_2d import NanodetLearner, draw_bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model that config file will be used", type=str, default='m')
    args = parser.parse_args()

    nanodet = NanodetLearner(model_to_use=args.model, device=args.device)
    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_{}".format(args.model), verbose=True)

    nanodet.download("./predefined_examples", mode="images")
    # First read an openDR image from your dataset and run the optimizer:
    img = Image.open("./predefined_examples/000000000036.jpg")
    nanodet.optimize("./onnx/nanodet_{}".format(args.model), img, optimization="onnx")

    boxes = nanodet.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=nanodet.classes, show=True)
