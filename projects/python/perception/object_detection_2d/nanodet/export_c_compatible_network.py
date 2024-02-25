# Copyright 2020-2024 OpenDR European Project
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
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="m")
    parser.add_argument("--dynamic", help="Determines if the model runs with dynamic shape input or not",
                        action="store_true")

    args = parser.parse_args()

    nanodet = NanodetLearner(model_to_use=args.model, device=args.device)
    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_{}".format(args.model), verbose=True)

    nanodet.optimize_c_model("./c_compatible_jit/nanodet_{}".format(args.model), conf_threshold=0.35,
                             iou_threshold=0.6, nms_max_num=100, dynamic=args.dynamic, verbose=True)
    print("C compatible network was exported in directory ./c_compatible_jit/nanodet_{}".format(args.model))
