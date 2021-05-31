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


from opendr.perception.face_recognition.face_recognition_learner import FaceRecognitionLearner
import argparse
from os.path import join
from opendr.engine.datasets import ExternalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--backbone", help="Backbone to use (mobilefacenet, ir_50)", type=str, default='mobilefacenet')
    args = parser.parse_args()

    onnx, device, backbone = args.onnx, args.device, args.backbone

    recognizer = FaceRecognitionLearner(device=device, backbone=backbone, mode='backbone_only', batch_size=2)

    recognizer.download(path=".")
    recognizer.load(".")

    if onnx:
        recognizer.optimize()

    # Download a sample dataset
    recognizer.download(path=".", mode="test_data")
    dataset_path = join(".", "test_data", "images")
    eval_dataset = ExternalDataset(path=dataset_path, dataset_type='imagefolder')
    results_dict = recognizer.eval(eval_dataset, num_pairs=1000)

    print("Evaluation results = ", results_dict)
