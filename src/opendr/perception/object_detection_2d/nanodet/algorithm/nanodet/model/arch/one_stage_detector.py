# Copyright 2021 RangiLyu.
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

import torch
import torch.nn as nn

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone import build_backbone
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn import build_fpn
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head import build_head


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head(x)
        return x

    def inference(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            if hasattr(self, "fpn"):
                x = self.fpn(x)
            if hasattr(self, "head"):
                if hasattr(self.head, "forward_infer"):
                    x = self.head.forward_infer(x)
                else:
                    x = self.head(x)
        return x

    def set_dynamic(self, dynamic=False):
        self.backbone.dynamic = dynamic
        if hasattr(self, "fpn"):
            self.fpn.dynamic = dynamic
        if hasattr(self, "head"):
            self.head.dynamic = dynamic
        if hasattr(self, "aux_fpn"):
            self.aux_fpn.dynamic = dynamic
        if hasattr(self, "aux_head"):
            self.aux_head.dynamic = dynamic

    def set_inference_mode(self, inference_mode=False):
        self.backbone.inference_mode = inference_mode
        if hasattr(self, "fpn"):
            self.fpn.inference_mode = inference_mode
        if hasattr(self, "head"):
            self.head.inference_mode = inference_mode

    def forward_train(self, gt_meta):
        preds = self(gt_meta["img"])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
