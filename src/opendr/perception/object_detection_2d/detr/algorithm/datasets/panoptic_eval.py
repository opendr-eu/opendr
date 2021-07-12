
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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

import json
import os

import opendr.perception.object_detection_2d.detr.algorithm.util.misc as utils

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass


class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir)
        return None
