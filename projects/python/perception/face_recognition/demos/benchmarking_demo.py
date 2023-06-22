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
import time
from opendr.perception.face_recognition import FaceRecognitionLearner
import argparse
from os.path import join
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--backbone", help="Backbone to use (mobilefacenet, ir_50)", type=str, default='mobilefacenet')
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--create_new", help="Whether to create or load a database", type=bool, default=True)
    args = parser.parse_args()

    onnx, device, backbone = args.onnx, args.device, args.backbone
    nvml = False
    try:
        if 'cuda' in device:
            from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
            nvmlInit()
            nvml = True
    except ImportError:
        print('You can install pynvml to also monitor the allocated GPU memory')
        pass
    if nvml:
        info_before = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        info_before = info_before.used / 1024 ** 2
    recognizer = FaceRecognitionLearner(device=device, backbone=backbone, mode='backbone_only')
    recognizer.download(path=".")
    recognizer.load(path=".")
    if nvml:
        info_after = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        info_after = info_after.used / 1024 ** 2
    # Download one sample image
    recognizer.download(path=".", mode="test_data")
    recognizer.fit_reference("./test_data", ".", create_new=args.create_new)
    image_path = join(".", "test_data", "images", "1", "1.jpg")
    img = cv2.imread(image_path)

    if onnx:
        recognizer.optimize()

    fps_list = []

    print("Benchmarking...")
    for i in tqdm(range(100)):
        start_time = time.perf_counter()
        # Perform inference
        result = recognizer.infer(img)
        end_time = time.perf_counter()
        fps_list.append(1.0 / (end_time - start_time))
    print("Average FPS: %.2f" % (np.mean(fps_list)))
    if nvml:
        print("Memory allocated: %.2f MB " % (info_after - info_before))
