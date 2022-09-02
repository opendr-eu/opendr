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

import os
import torch

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.batch_process import stack_batch_img
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.collate import naive_collate
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.transform import Pipeline
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


class Predictor(object):
    def __init__(self, cfg, model, device="cuda"):
        self.cfg = cfg
        self.device = device

        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.repvgg\
                import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)

        self.model = model.to(device).eval()

        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)

    def inference(self, img, verbose=True):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta, verbose)
        return meta, results


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names
