# Copyright 2020-2023 OpenDR European Project
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
# General imports
import os
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL


# yolov5 imports
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # workaround for rate limit bug


class YOLOv5DetectorLearner(Learner):
    available_models = ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x',
                        'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'custom']

    def __init__(self, model_name, path=None, device='cuda', temp_path='.', force_reload=False):
        super(YOLOv5DetectorLearner, self).__init__(device=device, temp_path=temp_path)
        self.device = device
        self.model_directory = temp_path if path is None else path
        self.model_name = model_name

        default_dir = torch.hub.get_dir()
        torch.hub.set_dir(temp_path)

        # Downloading and loading the fine-tuned yolov5s model in trucks
        if model_name == 'yolov5s_trucks':
            self.download(path='./', mode="pretrained", verbose=True)
            self.model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=path,
                                        force_reload=force_reload)
        # Getting a generic model
        else:
            if model_name not in self.available_models:
                model_name = 'yolov5s'
                print('Unrecognized model name, defaulting to "yolov5s"')

            if path is None:
                self.model = torch.hub.load('ultralytics/yolov5:master', 'custom',
                                            f'{temp_path}/{model_name}',
                                            force_reload=force_reload)
            else:
                self.model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=path,
                                            force_reload=force_reload)
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

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "/perception/object_detection_2d/yolov5/",
                 model_name='yolov5s_finetuned_in_trucks.pt', img_name='truck1.jpg'):
        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options are: ["pretrained",
        "images"].
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str, optional
        :param mode: one of: ["pretrained", "images"], where "pretrained" downloads a pretrained
        network, "images" downloads example inference data
        :type mode: str, optional
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param url: URL to file location on FTP server
        :type url: str, optional
        :param model_name: the name of the model file to download, currently only supports `yolov5s_finetuned_in_trucks.pt`
        :type model_name: str, optional
        :param img_name: name of image in ftp server, available files are `truckX.jpg` for `X=1 to 10`
        :type img_name: str, optional
        """
        valid_modes = ["pretrained", "images"]
        if mode not in valid_modes:
            raise ValueError("Invalid mode. Currently, only 'pretrained' and 'images' mode is supported.")

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            model_path = os.path.join(path, model_name)
            if not os.path.exists(model_path):
                if verbose:
                    print("Downloading pretrained model...")
                file_url = os.path.join(url, "pretrained", model_name)
                urlretrieve(file_url, model_path)
                if verbose:
                    print(f"Downloaded model to {model_path}.")
            else:
                if verbose:
                    print("Model already exists.")
        elif mode == "images":
            image_path = os.path.join(path, img_name)
            if not os.path.exists(image_path):
                if verbose:
                    print("Downloading example image...")
                file_url = os.path.join(url, "images", img_name)
                urlretrieve(file_url, image_path)
                if verbose:
                    print(f"Downloaded example image to {image_path}.")
            else:
                if verbose:
                    print("Example image already exists.")
