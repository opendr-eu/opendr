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

"""
Mostly copy-paste from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
"""
import torch
from opendr.perception.object_detection_2d.detr.algorithm.util.box_ops import rescale_bboxes


@torch.no_grad()
def detect(m1_im, m2_im, transform, model, postprocessor, device, threshold, ort_session):
    dev = torch.device(device)

    # mean-std normalize the input image (batch-size: 1)
    m1_img = transform(m1_im).unsqueeze(0)
    m2_img = transform(m2_im).unsqueeze(0)

    # propagate through the pytorch model
    m1_img = m1_img.to(dev)
    m2_img = m2_img.to(dev)
    model.eval()
    outputs = model([m1_img, m2_img])

    # keep only predictions with threshold confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], m1_im.size, device)

    segmentations = []

    sensor_contrib = outputs['auxiliary_test']
    return probas[keep], bboxes_scaled, segmentations, sensor_contrib
