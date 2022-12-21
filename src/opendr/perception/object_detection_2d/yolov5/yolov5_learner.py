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

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList

# yolov5 imports
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # workaround for rate limit bug


class YOLOv5DetectorLearner(Learner):
    available_models = ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x',
                        'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'custom']

    def __init__(self, model_name, path=None, device='cuda', temp_path='.', force_reload=False):
        super(YOLOv5DetectorLearner, self).__init__(device=device, temp_path=temp_path)
        if model_name not in self.available_models:
            model_name = 'yolov5s'
            print('Unrecognized model name, defaulting to "yolov5s"')

        default_dir = torch.hub.get_dir()
        torch.hub.set_dir(temp_path)

        if path is None:
            self.model = torch.hub.load('ultralytics/yolov5:master', 'custom', f'{temp_path}/{model_name}',
                                        force_reload=force_reload)
        else:
            self.model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=path,
                                        force_reload=force_reload, skip_validation=True)
        torch.hub.set_dir(default_dir)

        self.model.to(device)
        self.classes = [self.model.names[i] for i in range(len(self.model.names.keys()))]

    def infer(self, img, size=640):
        if isinstance(img, Image):
            img = img.convert("channels_last", "rgb")

        results = self.model(img, size=size)

        bounding_boxes = BoundingBoxList([])
        for idx, box in enumerate(results.xyxy[0]):
            box = box.cpu().numpy()
            bbox = BoundingBox(left=box[0], top=box[1],
                               width=box[2] - box[0],
                               height=box[3] - box[1],
                               name=box[5],
                               score=box[4])
            bounding_boxes.data.append(bbox)
        return bounding_boxes

    def fit(self):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def eval(self):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def load(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def save(self):
        """This method is not used in this implementation."""
        return NotImplementedError
