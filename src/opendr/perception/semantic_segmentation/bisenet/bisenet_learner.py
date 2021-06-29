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

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import tqdm
import numpy as np
import cv2
from imgaug import augmenters as iaa
from PIL import Image as PILImage
from urllib.request import urlretrieve
from opendr.perception.semantic_segmentation.bisenet.algorithm.model.build_BiSeNet import BiSeNet
from opendr.perception.semantic_segmentation.bisenet.algorithm.utils import reverse_one_hot, compute_global_accuracy, \
    fast_hist, per_class_iu, get_label_info, colour_code_segmentation
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.target import Heatmap


class BisenetLearner(Learner):
    def __init__(self,
                 lr=0.01,
                 iters=1,
                 batch_size=2,
                 optimizer='sgd',
                 device='cpu',
                 context_path='resnet18',
                 temp_path='',
                 checkpoint_after_iter=0,
                 checkpoint_load_iter=0,
                 val_after=1,
                 pin_memory=False,
                 num_classes=12,
                 weight_decay=5e-4,
                 drop_last=True,
                 momentum=0.9,
                 crop_height=720,
                 crop_width=960,
                 num_workers=4):
        super(BisenetLearner, self).__init__(lr=lr, batch_size=batch_size, iters=iters, optimizer=optimizer,
                                             temp_path=temp_path,
                                             checkpoint_after_iter=checkpoint_after_iter,
                                             checkpoint_load_iter=checkpoint_load_iter, device=device)

        self.context_path = context_path
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.drop_last = drop_last
        self.val_after = val_after
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.build_model()

        if self.optimizer == "sgd":
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum)
        elif self.optimizer == 'rmsprop':
            self.optimizer_func = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'adam':
            self.optimizer_func = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self):
        self.model = BiSeNet(num_classes=self.num_classes, context_path=self.context_path)
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model).cuda()

    def fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=True):
        """
        This method is used for training the algorithm on a train dataset

        """

        dataloader_train = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=self.drop_last)

        step = 0
        for epoch in range(1, self.iters + 1):
            self.model.train()
            tq = tqdm.tqdm(total=len(dataloader_train) * self.batch_size)
            tq.set_description('epoch %d, lr %f' % (epoch, self.lr))
            loss_record = []
            for i, (data, label) in enumerate(dataloader_train):
                if self.device == 'cuda':
                    data = data.cuda()
                    label = label.cuda()
                output, output_sup1, output_sup2 = self.model(data)
                loss1 = self.loss_func(output, label)
                loss2 = self.loss_func(output_sup1, label)
                loss3 = self.loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
                tq.update(self.batch_size)
                tq.set_postfix(loss='%.6f' % loss)
                self.optimizer_func.zero_grad()
                loss.backward()
                self.optimizer_func.step()
                step += 1
                loss_record.append(loss.item())
            tq.close()
            loss_train_mean = np.mean(loss_record)
            if not silent and verbose:
                print('Total Loss : %f' % (loss_train_mean))
            if self.checkpoint_after_iter != 0 and epoch % self.checkpoint_after_iter == 0:
                self.save(os.path.join(self.temp_path, 'epoch', str(epoch)))
            if self.val_after != 0 and epoch % self.val_after == 0 and val_dataset is not None:
                if not silent and verbose:
                    print('Validation...')
                self.eval(val_dataset)

    def eval(self, dataset, silent=False, verbose=True):

        dataloader_test = DataLoader(dataset,
                                     shuffle=False,
                                     pin_memory=self.pin_memory,
                                     drop_last=False,
                                     num_workers=self.num_workers)

        with torch.no_grad():
            self.model.eval()
            precision_record = []
            tq = tqdm.tqdm(total=len(dataloader_test) * 1)
            tq.set_description('test')
            hist = np.zeros((self.num_classes, self.num_classes))
            for i, (data, label) in enumerate(dataloader_test):
                tq.update(1)
                if self.device == 'cuda':
                    data = data.cuda()
                    label = label.cuda()
                predict = self.model(data).squeeze().cpu()
                predict = reverse_one_hot(predict)
                predict = np.array(predict)
                label = label.squeeze().cpu()
                label = np.array(label)
                precision = compute_global_accuracy(predict, label)
                hist += fast_hist(label.flatten(), predict.flatten(), self.num_classes)
                precision_record.append(precision)
            precision = np.mean(precision_record)
            miou_list = per_class_iu(hist)[:-1]
            miou = np.mean(miou_list)
            tq.close()
            if not silent and verbose:
                print('precision for test: %.3f' % precision)
                print('mIoU for validation: %.3f' % miou)

return {'precision': precision, 'miou': miou}

    def infer(self, img, csvpath, spath):
        """
        This method is used to perform semantic segmentation on an image.
        It returns a heatmap of the given image.

        """

        if not isinstance(img, Image):
            img = Image(img)
        img = img.numpy()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = iaa.Scale({'height': self.crop_height, 'width': self.crop_width})
        resize_det = resize.to_deterministic()
        image = resize_det.augment_image(image)
        image = PILImage.fromarray(image).convert('RGB')
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
        # read csv label path
        label_info = get_label_info(os.path.join(csvpath, 'class_dict.csv'))
        # predict
        if self.model is None:
            raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
        self.model.eval()
        predict = self.model(image).squeeze().cpu()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict), label_info)
        heatmap = Heatmap(predict)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(np.uint8(heatmap), (960, 720))
        sspath = os.path.join(spath, 'heatmap_example.png')
        cv2.imwrite(sspath, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        return heatmap

    def download(self, path=None, mode="pretrained", verbose=True,
                 url=OPENDR_SERVER_URL + "perception/semantic_segmentation/bisenet/"):
        """
        Download utility for various Semantic Segmentation components.
        Downloads files depending on mode and
        saves them in the path provided. It supports downloading:
        1)  pretrained model
        3) testing images

        :param path: Local path to save the files, defaults to self.temp_path if None
        :type path: str, path, optional
        :param mode:  "pretrained", "images", defaults to "pretrained"
        :type mode: str, optional
        :param verbose: Whether to print messages in the console, defaults to False
        :type verbose: bool, optional
        """
        valid_modes = ["pretrained", "testingImage"]

        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", should be one of:", valid_modes)

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            file_url = os.path.join(url, "trainedModels", "bisenet_camvid.pth")

            if verbose:
                print("Downloading pretrained weights...")
            file_path = os.path.join(path, "bisenet_camvid.pth")

            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

            file_url = os.path.join(url, "trainedModels", "bisenet_camvid.json")

            if verbose:
                print("Downloading json file...")
            file_path = os.path.join(path, "bisenet_camvid.json")

            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

        if mode == "testingImage":
            file_url = os.path.join(url, "datasets", "testImages", "test1.png")
            if verbose:
                print("Downloading a testing image...")
            file_path = os.path.join(path, "test1.png")

            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

    def save(self, path, verbose=True):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        model_name = os.path.basename(path)
        model_path = os.path.join(path, model_name + ".pth")
        if verbose:
            print(model_name)
        metadata = {"model_paths": [],
                    "framework": "pytorch",
                    "format": "pth",
                    "has_data": False,
                    "inference_params": {},
                    "optimized": False,
                    "optimizer_info": {}}
        param_filepath = model_name + ".pth"
        metadata["model_paths"].append(param_filepath)
        if self.device == 'cpu':
            torch.save(self.model.state_dict(), model_path)
        elif self.device == 'cuda':
            torch.save(self.model.module.state_dict(), model_path)
        if verbose:
            print("Model parameters saved.")

        with open(os.path.join(path, model_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if verbose:
            print("Model metadata saved.")

        return True

    def load(self, path):

        if self.model is None:
            self.build_model()

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Could not find directory {path}")

        folder_basename = os.path.basename(path)
        with open(os.path.join(path, folder_basename + ".json")) as jsonfile:
            metadata = json.load(jsonfile)

        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(os.path.join(path, metadata["model_paths"][0])))
        elif self.device == 'cuda':
            self.model.module.load_state_dict(torch.load(os.path.join(path, metadata["model_paths"][0])))
        self.model.eval()

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError
