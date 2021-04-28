# Copyright 2020-2021 OpenDR European Project
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
import time
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
import argparse
from os.path import join
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    args = parser.parse_args()

    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Download one sample image
    pose_estimator.download(path=".", mode="test_data")
    image_path = join("temp", "dataset", "image", "000000000785.jpg")
    img = cv2.imread(image_path)

    if onnx:
        pose_estimator.optimize()

    fps_list = []
    print("Benchmarking...")
    for i in tqdm(range(50)):
        start_time = time.perf_counter()
        # Perform inference
        poses = pose_estimator.infer(img)
        end_time = time.perf_counter()
        fps_list.append(1.0 / (end_time - start_time))
    print("Average FPS: %.2f" % (np.mean(fps_list)))

    # If pynvml is available, try to get memory stats for cuda
    try:
        if 'cuda' in device:
            from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex

            nvmlInit()
            info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
            print("Memory allocated: %.2f MB " % (info.used / 1024 ** 2))
    except ImportError:
        pass
