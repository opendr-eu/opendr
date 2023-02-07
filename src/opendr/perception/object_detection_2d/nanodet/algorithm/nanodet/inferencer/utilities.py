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


class Predictor(nn.Module):
    def __init__(self, cfg, model, device="cuda", nms_max_num=100):
        super(Predictor, self).__init__()
        self.cfg = cfg
        self.device = device
        self.nms_max_num = nms_max_num
        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.repvgg\
                import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)

        self.model = model.to(device).eval()

        for para in self.model.parameters():
            para.requires_grad = False

        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)
        self.traced_model = None

    def trace_model(self, dummy_input):
        self.traced_model = torch.jit.trace(self, dummy_input)
        return True

    def script_model(self, img, height, width, warp_matrix):
        preds = self.traced_model(img, height, width, warp_matrix)
        scripted_model = self.postprocessing(preds, img, height, width, warp_matrix)
        return scripted_model

    def forward(self, img, height=torch.tensor(0), width=torch.tensor(0), warp_matrix=torch.tensor(0)):
        if torch.jit.is_scripting():
            return self.script_model(img, height, width, warp_matrix)
        # In tracing (Jit and Onnx optimizations) we must first run the pipeline before the graf,
        # cv2 is needed, and it is installed with abi cxx11 but torch is in cxx<11
        meta = {"img": img}
        meta["img"] = divisible_padding(meta["img"], divisible=torch.tensor(32))
        with torch.no_grad():
            results = self.model.inference(meta)
        return results

    def preprocessing(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)

        _input = meta["img"]
        _height = torch.tensor(height)
        _width = torch.tensor(width)
        _warp_matrix = torch.from_numpy(meta["warp_matrix"])

        return _input, _height, _width, _warp_matrix

    def postprocessing(self, preds, input, height, width, warp_matrix):
        meta = {"height": height, "width": width, 'img': input, 'warp_matrix': warp_matrix}
        meta["img"] = divisible_padding(meta["img"], divisible=torch.tensor(32))
        res = self.model.head.post_process(preds, meta, nms_max_num=self.nms_max_num)
        return res
