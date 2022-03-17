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


from opendr.engine.datasets import ExternalDataset
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner

# TODO Sample of dataset should be downloaded from FTP
ur_dataset = ExternalDataset(path="./UR Fall Dataset", dataset_type="ur_fall_dataset")

device = "cuda"
stride = False
stages = 2
half_precision = False

pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                            mobilenet_use_stride=stride, half_precision=half_precision)
pose_estimator.download(path="./", verbose=True)
pose_estimator.load("openpose_default")

fall_detector = FallDetectorLearner(pose_estimator)

r = fall_detector.eval(dataset=ur_dataset)
print(f"Accuracy: {r['accuracy']} \nSensitivity: {r['sensitivity']} \nSpecificity: {r['specificity']} "
      f"\nDetection Accuracy: {r['detection_accuracy']} \nNo Detection Frames: {r['no_detections']}")
