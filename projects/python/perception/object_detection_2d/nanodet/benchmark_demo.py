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
from opendr.perception.object_detection_2d import NanodetLearner
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import draw_bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="m")
    parser.add_argument("--optimize-jit", help="", default=False, action="store_true")
    parser.add_argument("--optimize-onnx", help="", default=False, action="store_true")
    parser.add_argument("--repetitions", help="Determines the amount of repetitions to run", type=int, default=1000)
    parser.add_argument("--warmup", help="Determines the amount of warmup runs", type=int, default=100)
    parser.add_argument("--nms", help="Determines the max amount of bboxes the nms will output", type=int, default=30)
    parser.add_argument("--path", help="Path to the image that is used for inference", type=str,
                        default="./predefined_examples/000000000036.jpg")
    args = parser.parse_args()

    nanodet = NanodetLearner(model_to_use=args.model, device=args.device)
    nanodet.download("./predefined_examples", mode="pretrained", verbose=False)
    nanodet.load("./predefined_examples/nanodet_{}".format(args.model), verbose=False)
    nanodet.download("./predefined_examples", mode="images", verbose=False)

    img = Image.open(args.path)


    if args.optimize_jit:
        nanodet.optimize(f"./jit/nanodet_{args.model}", optimization="jit", verbose=False)
    if args.optimize_onnx:
        nanodet.optimize(f"./onnx/nanodet_{args.model}", optimization="onnx", verbose=False)

    nanodet.benchmark(img, repetitions=args.repetitions, warmup=args.warmup, nms_max_num=args.nms, half_precision=True)
