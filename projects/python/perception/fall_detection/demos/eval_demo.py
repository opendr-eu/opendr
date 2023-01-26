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

from opendr.engine.datasets import ExternalDataset
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use for pose estimator (cpu, cuda)", type=str, default="cuda")
    args = parser.parse_args()

    pose_estimator = LightweightOpenPoseLearner(device=args.device, num_refinement_stages=2,
                                                mobilenet_use_stride=False,
                                                half_precision=False)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    fall_detector = FallDetectorLearner(pose_estimator)

    # Download a sample dataset
    fall_detector.download(".", verbose=True)
    ur_dataset = ExternalDataset(path="./test_images", dataset_type="test")

    r = fall_detector.eval(dataset=ur_dataset, verbose=True)
    print(f"Evaluation results: \n Accuracy: {r['accuracy']} \n Sensitivity: {r['sensitivity']} "
          f"\n Specificity: {r['specificity']} \n Detection Accuracy: {r['detection_accuracy']} "
          f"\n No Detection Frames: {r['no_detections']}")
