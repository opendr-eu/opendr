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

import torch
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
import torch.optim as optim
import shutil
import ntpath
import os
import json
import onnxruntime as ort
import numpy as np
import torch.nn.functional as F
from urllib.request import urlretrieve
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2

from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.engine.datasets import ExternalDataset
from opendr.perception.binary_high_resolution.utils.architectures import VGG_720p_64, VGG_1080p_64
from opendr.perception.binary_high_resolution.utils.high_resolution_loader import HighResolutionDataset
from opendr.engine.constants import OPENDR_SERVER_URL


class BinaryHighResolutionLearner(Learner):
    def __init__(self, lr=1e-3, iters=100, batch_size=512, optimizer='adam', temp_path='', device='cpu',
                 weight_decay=1e-5, momentum=0.9, num_workers=4,
                 architecture='VGG_720p'):
        super(BinaryHighResolutionLearner, self).__init__(lr=lr, batch_size=batch_size, iters=iters,
                                                          optimizer=optimizer,
                                                          temp_path=temp_path, device=device)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_workers = num_workers
        self.architecture = architecture
        self.ort_session = None
        self._build_model()

    def _build_model(self):
        """
        This function initialized the PyTorch model based on the selected architecture
        """
        if self.architecture == 'VGG_1080p':
            self.model = VGG_1080p_64()
        elif self.architecture == 'VGG_720p':
            self.model = VGG_720p_64()
        else:
            raise ValueError("Architecture not supported.")
        if 'cuda' in self.device:
            self.model = self.model.to(self.device)

    def fit(self, dataset, silent=False, verbose=True):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.
        :param dataset: object that holds the training dataset
        :type dataset: ExternalDataset class object
        :param silent: if set to True, disables all printing of training progress reports and other information
          to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :return: returns stats regarding the training loss
        :rtype: dict
        """

        if not isinstance(dataset, ExternalDataset):
            assert ValueError("Data expected in ExternalDataset format")
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer not supported.")

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset = HighResolutionDataset(dataset.path, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                   sampler=ImbalancedDatasetSampler(dataset),
                                                   num_workers=self.num_workers)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.iters) / 2, gamma=0.1)
        self.model.train()
        n_steps = 0
        training_loss = []
        while n_steps < self.iters:
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):

                optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device).squeeze()
                output = self.model(data)
                aux_loss = target * (output[:, 1] - 1) ** 2 + (1 - target) * (output[:, 0] - 1) ** 2 + (1 - target) * (
                    output[:, 1]) ** 2 + target * (output[:, 1]) ** 2
                loss = F.cross_entropy(output.squeeze(), target) + torch.mean(aux_loss)
                loss.backward()
                if verbose:
                    print("Iteration %d out of %d, loss: %6.5f" % (n_steps, self.iters, loss.item()))
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                training_loss.append(total_loss)
                if n_steps > self.iters:
                    break
                n_steps += 1
            if not silent and not verbose:
                print("Epoch loss %6.5f" % total_loss)
        return {"loss": training_loss}

    def eval(self, dataset, silent=False, verbose=True):
        """
         This method is used to evaluate a trained model on an evaluation dataset.
         :param dataset: object that holds the evaluation dataset.
         :type dataset: ExternalDataset class object
         :param silent: This option is not supported by this implementation.
         :type silent: bool, optional
         :param verbose: This option is not supported by this implementation.
         :type verbose: bool, optional
         :returns: returns stats regarding evaluation
         :rtype: dict
         """
        if not isinstance(dataset, ExternalDataset):
            assert ValueError("Data expected in ExternalDataset format")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = HighResolutionDataset(dataset.path, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.model.eval()
        labels = []
        predictions = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device).squeeze()
            output = self.model(data).squeeze()
            output = F.softmax(output, dim=1).argmax(1)
            labels.extend(target.cpu().detach().numpy())
            predictions.append(output.cpu().detach().numpy())
        labels = np.asarray(labels).reshape((-1,))
        predictions = np.asarray(predictions).reshape((-1,))
        return {"precision": precision_score(labels, predictions), 'recall': recall_score(labels, predictions),
                'f1': f1_score(labels, predictions)}

    def infer(self, img):
        if self.model is None and self.ort_session is None:
            raise AttributeError("No model is loaded, cannot run inference. Load a model first using load().")

        if not isinstance(img, Image):
            img = Image(img)
        img_torch = torch.tensor(img.data).to(self.device).unsqueeze(0)
        if self.ort_session is not None:
            heatmap = self.ort_session.run(None, {'data': np.array(img_torch.cpu(), dtype=np.float32) / 255.0})
            heatmap = torch.tensor(heatmap[0])
        else:
            heatmap = self.model(img_torch / 255.0)
        heatmap = F.softmax(heatmap, dim=1)
        heatmap = heatmap.squeeze(0).cpu().detach().numpy()
        heatmap = heatmap[1, :, :]
        heatmap = cv2.resize(heatmap, (img.data.shape[2], img.data.shape[1]))
        heatmap = Heatmap(heatmap)
        return heatmap

    def download(self, path="./demo_dataset", verbose=False,
                 url=OPENDR_SERVER_URL + "perception/binary_high_resolution/demo_dataset/"):
        """
        Download utility for the toy dataset of the Binary High Resolution tool.

        :param path: Local path to save the files, defaults to self.temp_path if None
        :type path: str, path, optional
        :param verbose: Whether to print messages in the console, defaults to False
        :type verbose: bool, optional
        :param url: URL of the FTP server, defaults to OpenDR FTP URL
        :type url: str, optional
        """
        path = os.path.join(self.temp_path, path)

        if not os.path.exists(path):
            os.makedirs(path)

        if verbose:
            print("Downloading test data...")
        if not os.path.exists(path):
            os.makedirs(path)
        # Download annotation file
        file_url = os.path.join(url, "test_img.xml")
        if not os.path.exists(file_url):
            urlretrieve(file_url, os.path.join(path, "test_img.xml"))
        # Download test image
        file_url = os.path.join(url, "test_img.png")
        if not os.path.exists(file_url):
            urlretrieve(file_url, os.path.join(path, "test_img.png"))

        if verbose:
            print("Test data download complete.")

    def save(self, path, verbose=False):
        """
        This method is used to save a trained model.
        Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
        of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.
        If self.optimize was ran previously, it saves the optimized ONNX model in a similar fashion, by copying it
        from the self.temp_path it was saved previously during conversion.
        :param path: for the model to be saved, including the folder name
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        if self.model is None and self.ort_session is None:
            raise UserWarning("No model is loaded, cannot save.")

        folder_name, _, tail = self.__extract_trailing(path)  # Extract trailing folder name from path
        # Also extract folder name without any extension if extension is erroneously provided
        folder_name_no_ext = folder_name.split(sep='.')[0]

        # Extract path without folder name, by removing folder name from original path
        path_no_folder_name = path.replace(folder_name, '')
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == '':
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        full_path_to_model_folder = path_no_folder_name + folder_name_no_ext
        os.makedirs(full_path_to_model_folder, exist_ok=True)

        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}, "backbone": self.backbone}

        if self.ort_session is None:
            model_metadata["model_paths"] = [folder_name_no_ext + ".pth"]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"
            model_metadata["architecture"] = self.architecture

            custom_dict = {'state_dict': self.model.state_dict()}
            torch.save(custom_dict, os.path.join(full_path_to_model_folder, model_metadata["model_paths"][0]))
            if verbose:
                print("Saved Pytorch model.")
        else:
            model_metadata["model_paths"] = [os.path.join(folder_name_no_ext + ".onnx")]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"
            model_metadata["architecture"] = self.architecture
            # Copy already optimized model from temp path
            shutil.copy2(os.path.join(self.temp_path, "onnx_model_temp.onnx"),
                         os.path.join(full_path_to_model_folder, model_metadata["model_paths"][0]))
            model_metadata["optimized"] = True
            if verbose:
                print("Saved ONNX model.")

        with open(os.path.join(full_path_to_model_folder, folder_name_no_ext + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)

    @staticmethod
    def __extract_trailing(path):
        """
        Extracts the trailing folder name or filename from a path provided in an OS-generic way, also handling
        cases where the last trailing character is a separator. Returns the folder name and the split head and tail.
        :param path: the path to extract the trailing filename or folder name from
        :type path: str
        :return: the folder name, the head and tail of the path
        :rtype: tuple of three strings
        """
        head, tail = ntpath.split(path)
        folder_name = tail or ntpath.basename(head)  # handle both a/b/c and a/b/c/
        return folder_name, head, tail

    def load(self, path, verbose=False):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        model_name, _, _ = self.__extract_trailing(path)  # Trailing folder name from the path provided

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)

        if self.architecture != metadata['architecture']:
            raise ValueError(
                "Architectures do not match! Expected %s but got %s" % (self.architecture, metadata["architecture"]))

        if not metadata["optimized"]:
            self._build_model()
            checkpoint = torch.load(os.path.join(path, metadata["model_paths"][0]),
                                    map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint["state_dict"])
            if "cuda" in self.device:
                self.model.to(self.device)

            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.ort_session = ort.InferenceSession(os.path.join(path, metadata['model_paths'][0]))

            if verbose:
                print("Loaded ONNX model.")

    def optimize(self, do_constant_folding=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        if self.model is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")
        if self.ort_session is not None:
            raise UserWarning("Model is already optimized in ONNX.")

        try:
            self.__convert_to_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding)
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(self.temp_path, exist_ok=True)
            self.__convert_to_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding)
        self.ort_session = ort.InferenceSession(os.path.join(self.temp_path, "onnx_model_temp.onnx"))

    def __convert_to_onnx(self, output_name, do_constant_folding=False, verbose=False):
        """
        Converts the loaded regular PyTorch model to an ONNX model and saves it to disk.
        :param output_name: path and name to save the model, e.g. "/models/onnx_model.onnx"
        :type output_name: str
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        if "cuda" in self.device:
            inp = torch.randn(1, 3, 64, 64).to(self.device)
        else:
            inp = torch.randn(1, 3, 64, 64)
        input_names = ['data']
        output_names = ['output']

        torch.onnx.export(self.model, inp, output_name, verbose=verbose, opset_version=11,
                          do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names,
                          dynamic_axes={"data": {0: "batch", 2: "height", 3: "width"}})

    def reset(self):
        """This method is not used in this implementation."""
        raise NotImplementedError
