# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
"""
import torch
from opendr.perception.object_detection_2d.detr.algorithm.util.box_ops import rescale_bboxes
import numpy as np

@torch.no_grad()
def detect(im, transform, model, device, threshold, ort_session):
        dev = torch.device(device)

        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        if ort_session is not None:
            # propagate through the onnx model
            outputs = ort_session.run(['pred_logits', 'pred_boxes'], {'data': np.array(img)})

            pred_logits = torch.tensor(outputs[0], device=dev)
            pred_boxes = torch.tensor(outputs[1], device=dev)

            # keep only predictions with threshold confidence
            probas = pred_logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold

            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], im.size, device)
            return probas[keep], bboxes_scaled
        else:
            # propagate through the pytorch model
            img = img.to(dev)
            model.eval()
            outputs = model(img)

            # keep only predictions with threshold confidence
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold

            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
            return probas[keep], bboxes_scaled
