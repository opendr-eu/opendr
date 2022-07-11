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

from opendr.perception.object_detection_2d import NanodetLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--image-path", help="Path to images for inference", type=str)
    parser.add_argument("--model", help="Model that config file will be used", type=str)
    parser.add_argument("--time", type=int, default=1)
    args = parser.parse_args()

    nanodet = NanodetLearner(config=args.model, device=args.device)
    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet-{}/nanodet-{}.ckpt".format(args.model, args.model), verbose=True)
    boxes = nanodet.infer(path=args.image_path, time_to_open=args.time)

