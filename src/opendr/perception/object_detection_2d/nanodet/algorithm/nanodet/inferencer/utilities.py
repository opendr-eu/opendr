# Modifications Copyright 2021 - present, OpenDR European Project
#
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

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.batch_process import divisible_padding
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.transform import Pipeline
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model


class ScriptedPredictor(nn.Module):
    def __init__(self, model, dummy_input, conf_thresh=0.35, iou_thresh=0.6, nms_max_num=100, dynamic=False):
        super(ScriptedPredictor, self).__init__()
        model.forward = model.inference
        self.model = model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.nms_max_num = nms_max_num
        self.jit_model = torch.jit.script(self.model) if dynamic else torch.jit.trace(self.model, dummy_input[0])

    def forward(self, input, height, width, warp_matrix):
        preds = self.jit_model(input)
        meta = dict(height=height, width=width, warp_matrix=warp_matrix, img=input)
        return self.model.head.post_process(preds, meta, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
                                            nms_max_num=self.nms_max_num)


class Predictor(nn.Module):
    def __init__(self, cfg, model, device="cuda", conf_thresh=0.35, iou_thresh=0.6, nms_max_num=100,
                 hf=False, dynamic=False, ch_l=False):
        super(Predictor, self).__init__()
        self.cfg = cfg
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.nms_max_num = nms_max_num
        self.hf = hf
        self.ch_l = ch_l
        self.dynamic = dynamic and self.cfg.data.val.keep_ratio
        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.repvgg\
                import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)

        for para in model.parameters():
            para.requires_grad = False

        if self.ch_l:
            model = model.to(memory_format=torch.channels_last)
        if self.hf:
            model = model.half()
        model.set_dynamic(self.dynamic)
        model.set_inference_mode(True)

        self.model = model.to(device).eval()

        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)

    def trace_model(self, dummy_input):
        return torch.jit.trace(self, dummy_input[0])

    def script_model(self):
        return torch.jit.script(self)

    def c_script(self, dummy_input):
        import copy
        jit_ready_predictor = ScriptedPredictor(copy.deepcopy(self.model), dummy_input, self.conf_thresh,
                                                self.iou_thresh, self.nms_max_num, dynamic=self.dynamic)
        return torch.jit.script(jit_ready_predictor)

    def forward(self, img):
        return self.model.inference(img)

    def preprocessing(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)

        meta["img"] = divisible_padding(
            meta["img"],
            divisible=torch.tensor(32, device=self.device)
        )

        _input = meta["img"].to(torch.half if self.hf else torch.float32)
        _input = _input.to(memory_format=torch.channels_last) if self.ch_l else _input
        _height = torch.as_tensor(height, device=self.device)
        _width = torch.as_tensor(width, device=self.device)
        _warp_matrix = torch.from_numpy(meta["warp_matrix"]).to(self.device)

        return _input, _height, _width, _warp_matrix

    def postprocessing(self, preds, input, height, width, warp_matrix):
        meta = dict(height=height, width=width, warp_matrix=warp_matrix, img=input)
        res = self.model.head.post_process(preds, meta, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
                                           nms_max_num=self.nms_max_num)
        return res
