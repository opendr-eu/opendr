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
from opendr.perception.object_detection_2d import NanodetLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    config = "/home/manos/new_opendr/opendr/src/opendr/perception/object_detection_2d/nanodet/algorithm/config/nanodet-plus-m_416.yml"
    load_path = "/home/manos/new_opendr/opendr/src/opendr/perception/object_detection_2d/nanodet/algorithm/pretrained_models/nanodet-plus-m_416_checkpoint.ckpt"
    nanodet = NanodetLearner(config=config)
    # ssd.download(".", mode="pretrained")
    nanodet.load(load_path, verbose=True)

    # ssd.download(".", mode="images")

    # dataset_root = "/home/manos/data/coco2017/val2017"
    # image_path = "{}/000000000724.jpg".format(dataset_root)
    # boxes = nanodet.infer(path=image_path)

    dataset_root = "/home/manos/Downloads"
    video_path = "{}/temp_video.mp4".format(dataset_root)
    boxes = nanodet.infer(path=video_path, mode="video")

