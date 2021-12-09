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

# MIT License
#
# Copyright (c) 2019 Jian Zhao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from PIL import Image as PILImage
import numpy as np
import pickle
import cv2
import onnxruntime as ort
from tqdm import tqdm
import os
import json
import shutil
from urllib.request import urlretrieve

from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.perception.face_recognition.algorithm.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from opendr.perception.face_recognition.algorithm.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, \
    IR_SE_152
from opendr.perception.face_recognition.algorithm.backbone.model_mobilenet import MobileFaceNet
from opendr.perception.face_recognition.algorithm.head.losses import ArcFace, CosFace, SphereFace, AMSoftmax, Classifier
from opendr.perception.face_recognition.algorithm.loss.focal import FocalLoss
from opendr.perception.face_recognition.algorithm.util.utils import make_weights_for_balanced_classes, get_val_data, \
    separate_irse_bn_paras, separate_mobilenet_bn_paras, l2_norm, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, perform_val_imagefolder, buffer_val, AverageMeter, \
    accuracy
from opendr.perception.face_recognition.algorithm.align.align import face_align


class FaceRecognitionLearner(Learner):
    def __init__(self, lr=0.1, iters=120, batch_size=128, optimizer='sgd', device='cuda', threshold=0.0,
                 backbone='ir_50', network_head='arcface', loss='focal',
                 temp_path='./temp', mode='backbone_only',
                 checkpoint_after_iter=0, checkpoint_load_iter=0, val_after=0,
                 input_size=[112, 112], rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5], embedding_size=512,
                 weight_decay=5e-4, momentum=0.9, drop_last=True, stages=[35, 65, 95],
                 pin_memory=True, num_workers=4,
                 seed=123):
        super(FaceRecognitionLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                                     backbone=backbone, network_head=network_head, temp_path=temp_path,
                                                     checkpoint_after_iter=checkpoint_after_iter,
                                                     checkpoint_load_iter=checkpoint_load_iter,
                                                     device=device, threshold=threshold)

        if input_size is None:
            input_size = [112, 112]
        if rgb_mean is None:
            rgb_mean = [0.5, 0.5, 0.5]
        if rgb_std is None:
            rgb_std = [0.5, 0.5, 0.5]
        if stages is None:
            stages = [35, 65, 95]
        self.seed = seed
        self.loss = loss
        self.mode = mode
        self.input_size = input_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        if self.backbone == 'mobilefacenet':
            self.embedding_size = 128
        else:
            self.embedding_size = embedding_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.drop_last = drop_last
        self.stages = stages
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        torch.manual_seed(self.seed)

        self.network_head_model = None
        self.backbone_model = None
        self.epoch = checkpoint_load_iter

        self._model = None
        self.writer = None
        self.logging = False
        self.database = None
        self.num_class = 0
        self.classes = None
        self.opt = None
        self.val_after = val_after
        self.data = None
        self.pairs = None
        self.ort_backbone_session = None  # ONNX runtime inference session for backbone
        self.ort_head_session = None  # ONNX runtime inference session for head
        self.temp_path = temp_path

    def __create_model(self, num_class=0):
        # Create the backbone architecture
        self.num_class = num_class
        if self.backbone_model is None:
            backbone_dict = {'resnet_50': ResNet_50(self.input_size),
                             'resnet_101': ResNet_101(self.input_size),
                             'resnet_152': ResNet_152(self.input_size),
                             'ir_50': IR_50(self.input_size),
                             'ir_101': IR_101(self.input_size),
                             'ir_152': IR_152(self.input_size),
                             'ir_se_50': IR_SE_50(self.input_size),
                             'ir_se_101': IR_SE_101(self.input_size),
                             'ir_se_152': IR_SE_152(self.input_size),
                             'mobilefacenet': MobileFaceNet()}
            backbone = backbone_dict[self.backbone]
            self.backbone_model = backbone.to(self.device)
        # Create the head architecture
        if self.mode != 'backbone_only':
            head_dict = {
                'arcface': ArcFace(in_features=self.embedding_size, out_features=self.num_class, device=self.device),
                'cosface': CosFace(in_features=self.embedding_size, out_features=self.num_class, device=self.device),
                'sphereface': SphereFace(in_features=self.embedding_size, out_features=self.num_class,
                                         device=self.device),
                'am_softmax': AMSoftmax(in_features=self.embedding_size, out_features=self.num_class,
                                        device=self.device),
                'classifier': Classifier(in_features=self.embedding_size, out_features=self.num_class,
                                         device=self.device)}
            head = head_dict[self.network_head]
            self.network_head_model = head.to(self.device)
        else:
            self.network_head_model = None

    def align(self, data='', dest='/aligned', crop_size=112, silent=False):
        """
        This method is used for aligning the faces in an imagefolder dataset.

        :param data: The folder containing the images to be aligned
        :type data: str
        :param dest: destination folder to save the aligned images, defaults to './temp/aligned'
        :type dest: str
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        """
        face_align(data, dest, crop_size)
        if not silent:
            print('Face align finished')

    def fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=True):
        """
        This method is used for training the algorithm on a train dataset and
        validating on a val dataset.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns stats regarding training and validation.

        :param dataset: Object that holds the training dataset
        :type dataset: Dataset class type
        :param val_dataset: Object that holds the validation dataset
        :type val_dataset: Dataset class type, optional
        :param verbose: if set to True, enables the maximum logging verbosity (depends on the actual algorithm)
        :type verbose: bool, optional
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        :param logging_path: path to save tensorboard log files. If set to None or ‘’, tensorboard logging is disabled
        :type logging_path: str, optional
        :return: Returns stats regarding training and validation
        :rtype: dict
        """
        loss_list = []
        acc_list = []
        eval_results = []
        # Tensorboard logging
        if logging_path != '':
            self.logging = True
            self.writer = SummaryWriter(logging_path)
        else:
            self.logging = False

        train_transform = transforms.Compose([
            transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
            transforms.RandomCrop([self.input_size[0], self.input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean,
                                 std=self.rgb_std),
        ])

        if dataset.dataset_type != 'imagefolder':
            raise UserWarning('dataset should be of type imagefolder')

        dataset_train = datasets.ImageFolder(dataset.path, train_transform)

        # Create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, sampler=sampler, pin_memory=self.pin_memory,
            num_workers=self.num_workers, drop_last=self.drop_last
        )

        loss_dict = {'focal': FocalLoss(),
                     'softmax': nn.CrossEntropyLoss()}
        criterion = loss_dict[self.loss]
        self.classes = train_loader.dataset.classes
        self.num_class = len(train_loader.dataset.classes)
        self.__create_model(self.num_class)
        self._model = {self.backbone_model, self.network_head_model}

        if self.checkpoint_load_iter != 0:
            if self.mode == 'head_only':
                if os.path.exists(
                        os.path.join(self.temp_path, 'checkpoints', 'head_{}_iter_{}'.format(
                            self.network_head, self.epoch))):
                    head_info = torch.load(os.path.join(self.temp_path, 'checkpoints', 'head_{}_iter_{}'.format(
                        self.network_head, self.epoch)))
                    self.network_head_model.load_state_dict(head_info['head_state_dict'])
                else:
                    raise UserWarning('No checkpoint  head_{}_iter_{} found'.format(
                        self.network_head, self.epoch))
            else:
                if os.path.exists(os.path.join(self.temp_path, 'checkpoints', 'backbone_{}_iter_{}'.format(
                        self.backbone, self.epoch))) and os.path.exists(
                    os.path.join(self.temp_path, 'checkpoints', 'head_{}_iter_{}'.format(
                        self.network_head, self.epoch))):
                    backbone_info = torch.load(os.path.join(self.temp_path, 'checkpoints', 'backbone_{}_iter_{}'.format(
                        self.backbone, self.epoch)))
                    self.backbone_model.load_state_dict(backbone_info['backbone_state_dict'])
                    head_info = torch.load(os.path.join(self.temp_path, 'checkpoints', 'head_{}_iter_{}'.format(
                        self.network_head, self.epoch)))
                    self.network_head_model.load_state_dict(head_info['head_state_dict'])
                else:
                    raise UserWarning('No correct checkpoint files found')

        # Separate batch_norm parameters from others
        # Do not do weight decay for batch_norm parameters to improve the generalizability
        if self.backbone.find("ir") >= 0:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(
                self.backbone_model)
            _, head_paras_wo_bn = separate_irse_bn_paras(self.network_head_model)
        elif self.backbone == 'mobilefacenet':
            _, head_paras_wo_bn = separate_mobilenet_bn_paras(self.network_head_model)
        else:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
                self.backbone_model)
            _, head_paras_wo_bn = separate_resnet_bn_paras(self.network_head_model)

        # ======= Train & validation & save checkpoint =======#
        disp_freq = len(train_loader) // 100  # Frequency to display training loss & acc
        if disp_freq < 1:
            disp_freq = 1000

        num_epoch_warm_up = self.iters // 25
        num_batch_warm_up = len(train_loader) * num_epoch_warm_up

        if self.mode == 'head_only':
            optimizer = optim.SGD([{'params': head_paras_wo_bn, 'weight_decay': self.weight_decay}], lr=self.lr,
                                  momentum=self.momentum)
        else:
            if self.backbone == 'mobilefacenet':
                optimizer = optim.SGD([{'params': list(self.backbone_model.parameters()) + head_paras_wo_bn,
                                        'weight_decay': self.weight_decay}],
                                      lr=self.lr, momentum=self.momentum)
            else:
                optimizer = optim.SGD(
                    [{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': self.weight_decay},
                     {'params': backbone_paras_only_bn}], lr=self.lr, momentum=self.momentum)
        self.opt = optimizer
        if self.checkpoint_load_iter != 0:
            self.opt.load_state_dict(head_info['optimizer_state_dict'])
        for epoch in range(self.iters):  # Start training process
            if self.epoch == self.stages[0]:  # Adjust LR for each training stage after warm up
                schedule_lr(self.opt)
            if self.epoch == self.stages[1]:
                schedule_lr(self.opt)
            if self.epoch == self.stages[2]:
                schedule_lr(self.opt)

            if self.mode == 'head_only':
                self.backbone_model.eval()
            else:
                self.backbone_model.train()
            self.network_head_model.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            self.network_head_model = self.network_head_model.to(self.device)
            batch = 0  # batch index
            for inputs, labels in tqdm(iter(train_loader), disable=(not verbose or silent)):
                if (epoch + 1 <= num_epoch_warm_up) and (
                        batch + 1 <= num_batch_warm_up):  # Adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, num_batch_warm_up, self.lr, optimizer)

                # Compute output
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                features = self.backbone_model(inputs)
                outputs = self.network_head_model(features, labels)
                loss = criterion(outputs, labels)

                # Measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # Compute gradient and do SGD step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Display training loss & acc every disp_freq
                if not silent and verbose:
                    if ((batch + 1) % disp_freq == 0) and batch != 0:
                        print("=" * 60)
                        print('Epoch {}/{} Batch {}/{}\t'
                              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                  self.epoch + 1, self.iters, batch + 1, len(train_loader), loss=losses,
                                  top1=top1, top5=top5))
                        print("=" * 60)

                batch += 1  # Batch index

            # Training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)
            if self.logging:
                self.writer.add_scalar("Training_Loss", epoch_loss, self.epoch + 1)
                self.writer.add_scalar("Training_Accuracy", epoch_acc, self.epoch + 1)
            if not silent:
                print("=" * 60)
                print('Epoch: {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          self.epoch + 1, self.iters, loss=losses, top1=top1, top5=top5))

                print("=" * 60)
            self.epoch += 1
            if self.val_after != 0:
                if self.epoch % self.val_after == 0:
                    eval_results.append(self.eval(val_dataset))
            if self.checkpoint_after_iter != 0:
                if self.epoch % self.checkpoint_after_iter == 0:
                    self.__save(os.path.join(self.temp_path, 'checkpoints'))

        results = {'Training_Loss': loss_list, 'Training_Accuracy': acc_list}

        return {'Training_statistics': results, 'Evaluation_statistics': eval_results}

    def fit_reference(self, path=None, save_path=None):
        """
        Implementation to create reference database. Provided with a path with reference images and a save_path,
        it creates a .pkl file containing the features of the reference images, and saves it
        in the save_path.
        :param path: path containing the reference images
        :type path: str
        :param save_path: path to save the .pkl file
        :type save_path: str
        """
        if self._model is None and self.ort_backbone_session is None:
            raise UserWarning('A model should be loaded first')
        if os.path.exists(os.path.join(save_path, 'reference.pkl')):
            print('Loading Reference')
            self.database = pickle.load(open(os.path.join(save_path, 'reference.pkl'), "rb"))
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            database = {}
            transform = transforms.Compose([
                transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
                transforms.CenterCrop([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)]
            )
            with torch.no_grad():
                class_key = 0
                for subdir, dirs, files in os.walk(path):
                    if subdir == path:
                        continue
                    total = 0
                    features_sum = torch.zeros(1, self.embedding_size).to(self.device)
                    for file in files:
                        total += 1
                        inputs = cv2.imread(os.path.join(subdir, file))
                        inputs = transform(PILImage.fromarray(
                            cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB))
                        )
                        inputs = inputs.unsqueeze(0)
                        if self.ort_backbone_session is not None:
                            features = self.ort_backbone_session.run(None, {'data': np.array(inputs.cpu())})
                            features = torch.tensor(features[0]).to(self.device)
                        else:
                            self.backbone_model.eval()
                            inputs = inputs.to(self.device)
                            features = self.backbone_model(inputs)
                        features = l2_norm(features)
                        features_sum += features
                    avg_features = features_sum / total
                    subdir = subdir.split('/')
                    subdir = subdir[-1]
                    database[class_key] = [subdir, avg_features]
                    class_key += 1
            f = open(os.path.join(save_path, "reference.pkl"), "wb")
            pickle.dump(database, f)
            f.close()
            self.database = database

    def infer(self, img):
        """
        This method is used to perform face recognition on an image.

        :param img: image to run inference on
        :rtype img: engine.data.Image class object
        :return: Returns an engine.target.Category object, which holds an ID and the distance between
                 the embedding of the input image and the closest embedding existing in the reference database.
        :rtype: engine.target.Category object
        """
        if not isinstance(img, Image):
            img = Image(img)
        img = img.convert("channels_last", "bgr")
        if self._model is None and self.ort_backbone_session is None:
            raise UserWarning('A model should be loaded first')
        transform = transforms.Compose([
            transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
            transforms.CenterCrop([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)]
        )
        if self.mode == 'backbone_only':
            distance = self.threshold
            if distance == 0:
                distance = 10
            person = None
            self.backbone_model.eval()
            with torch.no_grad():
                img = PILImage.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = transform(img)
                img = img.unsqueeze(0)
                img = img.to(self.device)
                if self.ort_backbone_session is not None:
                    features = self.ort_backbone_session.run(None, {'data': np.array(img.cpu())})
                    features = torch.tensor(features[0])
                else:
                    self.backbone_model.eval()
                    features = self.backbone_model(img)
                features = l2_norm(features)
            if self.database is None:
                raise UserWarning('A reference for comparison should be created first. Try calling fit_reference()')
            for key in self.database:
                diff = np.subtract(features.cpu().numpy(), self.database[key][1].cpu().numpy())
                dist = np.sum(np.square(diff), axis=1)
                if np.isnan(dist):
                    dist = 10
                if dist < distance:
                    distance = dist
                    person = key
            if type(distance) != float:
                confidence = 1 - (distance.item() / self.threshold)
            else:
                confidence = 1 - (distance / self.threshold)
            if person is not None:
                person = Category(person, self.database[person][0], confidence)
                return person
            else:
                person = Category(-1, 'Not found', 0.0)
                return person

        elif self.network_head == 'classifier':
            self.backbone_model.eval()
            self.network_head_model.eval()
            with torch.no_grad():
                img = transform(PILImage.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                )
                img = img.unsqueeze(0)
                img = img.to(self.device)
                if self.ort_backbone_session is not None:
                    features = self.ort_backbone_session.run(None, {'data': np.array(img.cpu())})
                    features = torch.tensor(features[0])
                else:
                    self.backbone_model.eval()
                    features = self.backbone_model(img)
                features = l2_norm(features)
                if self.ort_head_session is not None:
                    outs = self.ort_head_session.run(None, {'features': np.array(features.cpu())})
                    return self.classes[outs.index(max(outs))]
                else:
                    outs = self.network_head_model(features)
                    _, predicted = torch.max(outs.data, 1)
                    person = Category(self.classes[predicted.item()])
                    return person
        else:
            raise UserWarning('Infer should be called either with backbone_only mode or with a classifier head')

    def eval(self, dataset=None, num_pairs=1000, silent=False, verbose=True):
        """
        This method is used to evaluate a trained model on an evaluation dataset.

        :param dataset: object that holds the evaluation dataset.
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param num_pairs: the number of pairs to be created for evaluation in a custom imagefolder dataset,
                    defaults to 1000
        :type num_pairs: int
        :param silent: if set to True, disables all printing of evalutaion progress reports and other information
                       to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :returns: returns stats regarding evaluation
        :rtype: dict
        """
        if self._model is None:
            raise UserWarning('A model should be loaded first')
        if self.network_head != 'classifier' and self.mode != 'head_only':
            self.backbone_model.eval()
            if not silent:
                print("=" * 60)
                print("Perform Evaluation on " + dataset.dataset_type)
            if dataset.dataset_type == 'imagefolder':
                self.data = get_val_data(dataset.path, dataset.dataset_type, num_pairs)
                eval_accuracy, best_threshold = perform_val_imagefolder(self.device, self.embedding_size,
                                                                        self.batch_size, self.backbone_model, self.data,
                                                                        self.num_workers)
            else:
                self.data, self.pairs = get_val_data(dataset.path, dataset.dataset_type)
                eval_accuracy, best_threshold = perform_val(self.device, self.embedding_size,
                                                            self.batch_size, self.backbone_model, self.data,
                                                            self.pairs)

            self.threshold = float(best_threshold)
            if self.logging:
                buffer_val(self.writer, dataset.dataset_type, eval_accuracy, best_threshold, self.epoch + 1)
            if not silent:
                print(
                    "Evaluation on " + dataset.dataset_type + ": Acc: {} ".format(eval_accuracy))
                print("=" * 60)
            return {'Accuracy': eval_accuracy, 'Best_Threshold': best_threshold}

        else:
            self.backbone_model.eval()
            if self.mode != 'backbone_only':
                self.network_head_model.eval()
            if not silent:
                print("Perform Evaluation on Image Dataset")
            eval_transform = transforms.Compose([
                transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
                transforms.CenterCrop([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)]
            )

            dataset_eval = datasets.ImageFolder(os.path.join(dataset.path, dataset.dataset_type), eval_transform)
            eval_loader = torch.utils.data.DataLoader(
                dataset_eval, batch_size=self.batch_size, pin_memory=self.pin_memory,
                num_workers=self.num_workers, drop_last=self.drop_last
            )
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(iter(eval_loader)):
                    # compute output
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).long()
                    features = self.backbone_model(inputs)
                    outputs = self.network_head_model(features, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            if self.logging:
                self.writer.add_scalar("Evaluation_Accuracy", 100 * correct / total, self.epoch + 1)
            if not silent:
                print('Accuracy of the network on {} test images: {}'.format(total,
                                                                             100 * correct / total))
            return {"Accuracy": 100 * correct / total}

    def __save(self, path):
        """
        Internal save implementation is used to create checkpoints. Provided with a path,
        it adds training state information in a custom dict.
        :param path: path for the model to be saved
        :type path: str
        """
        if not os.path.exists(path):
            os.makedirs(path)
        backbone_custom_dict = {'backbone_state_dict': self.backbone_model.state_dict(),
                                'current_epoch': self.epoch}
        head_custom_dict = {'head_state_dict': self.network_head_model.state_dict(),
                            'num_class': self.num_class,
                            'classes': self.classes,
                            'current_epoch': self.epoch,
                            'optimizer_state_dict': self.opt.state_dict()}
        if self.mode == 'head_only':
            torch.save(head_custom_dict, os.path.join(path, "head_{}_iter_{}".format(
                self.network_head, self.epoch)))
        torch.save(backbone_custom_dict, os.path.join(path, "backbone_{}_iter_{}".format(
            self.backbone, self.epoch)))
        torch.save(head_custom_dict, os.path.join(path, "head_{}_iter_{}".format(
            self.network_head, self.epoch)))

    def download(self, path=None, mode="pretrained"):
        """
        Download utility for various Face Recognition components. Downloads files depending on mode and
        saves them in the path provided. It supports downloading:
        1) the corresponding pretrained backbone
        3) a test dataset with a single image

        :param path: Local path to save the files, defaults to self.temp_path if None
        :type path: str, path, optional
        :param mode: What file to download, can be one of "pretrained", "test_data", defaults to "pretrained"
        :type mode: str, optional
        :param verbose: Whether to print messages in the console, defaults to False
        :type verbose: bool, optional
        """
        valid_modes = ['pretrained', 'test_data']
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", should be one of:", valid_modes)
        if mode == 'pretrained':

            if path is None:
                path = self.temp_path
            if not os.path.exists(path):
                os.makedirs(path)

            if self.mode == 'backbone_only' or self.mode == 'finetune':
                if not os.path.exists(os.path.join(path, 'backbone_' + self.backbone + '.pth')):
                    url = OPENDR_SERVER_URL + 'perception/face_recognition/'
                    url_backbone = os.path.join(url, 'backbone_' + self.backbone + '.pth')
                    url_backbone_json = os.path.join(url, 'backbone_' + self.backbone + '.json')
                    urlretrieve(url_backbone, os.path.join(path, 'backbone_' + self.backbone + '.pth'))
                    urlretrieve(url_backbone_json, os.path.join(path, 'backbone_' + self.backbone + '.json'))
                    print('Model downloaded')
                else:
                    print('Model already exists')
            else:
                raise UserWarning('Only a pretrained backbone can be downloaded,'
                                  ' change Learners mode to "backbone_only" or "finetune"')

        if mode == 'test_data':

            if path is None:
                path = self.temp_path
            if not os.path.exists(path):
                os.makedirs(path)

            image_parent = os.path.join(path, 'test_data', 'images')
            if not os.path.exists(image_parent):
                os.makedirs(image_parent)
                url = OPENDR_SERVER_URL + 'perception/face_recognition/test_data/images'
                for i in range(1, 7):
                    image_path = os.path.join(image_parent, str(i))
                    ftp_parent = os.path.join(url, str(i))
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    for j in range(1, 3):
                        image_dl = os.path.join(ftp_parent, str(j) + '.jpg')
                        urlretrieve(image_dl, os.path.join(image_path, str(j) + '.jpg'))
                print('Data Downloaded')
            else:
                print('Data already downloaded')

    def save(self, path=None):
        """
        Save for external usage.
        This will be loaded with self.load.
        :param path for the model to be saved
        :type path: str
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if self.mode == 'backbone_only' or self.mode == 'finetune' or \
                (self.mode == 'full' and self.network_head != 'classifier'):
            self.__save_backbone(path)
        else:
            if self.mode == 'head_only':
                self.__save_head(path)
            elif self.mode == 'full' and self.network_head == 'classifier':
                self.__save_backbone(path)
                self.__save_head(path)

    def __save_backbone(self, path):
        if self.ort_backbone_session is None:
            backbone_metadata = {'model_paths': os.path.join(path, 'backbone_' + self.backbone + '.pth'),
                                 'framework': 'pytorch',
                                 'format': 'pth',
                                 'has_data': False,
                                 'inference_params': {'threshold': self.threshold},
                                 'optimized': False,
                                 'optimizer_info': {}
                                 }
            torch.save(self.backbone_model.state_dict(), os.path.join(path, 'backbone_' + self.backbone + '.pth'))
        else:
            backbone_metadata = {'model_paths': os.path.join(path, 'onnx_' + self.backbone + '_backbone_model.onnx'),
                                 'framework': 'pytorch',
                                 'format': 'onnx',
                                 'has_data': False,
                                 'inference_params': {'threshold': self.threshold},
                                 'optimized': True,
                                 'optimizer_info': {}
                                 }
            shutil.copy2(os.path.join(self.temp_path, 'onnx_' + self.backbone + '_backbone_model.onnx'),
                         backbone_metadata['model_paths'])
        with open(os.path.join(path, 'backbone_' + self.backbone + '.json'), 'w', encoding='utf-8') as f:
            json.dump(backbone_metadata, f, ensure_ascii=False, indent=4)

    def __save_head(self, path):
        if self.ort_head_session is None:
            head_metadata = {'model_paths': os.path.join(path, 'head_' + self.network_head + '.pth'),
                             'framework': 'pytorch',
                             'format': 'pth',
                             'has_data': False,
                             'inference_params': {'num_class': self.num_class,
                                                  'classes': self.classes},
                             'optimized': False,
                             'optimizer_info': {}
                             }

            torch.save(self.network_head_model.state_dict(),
                       os.path.join(path, 'head_' + self.network_head + '.pth'))
        else:
            head_metadata = {'model_paths': os.path.join(path, 'onnx_' + self.network_head + '_head_model.onnx'),
                             'framework': 'pytorch',
                             'format': 'onnx',
                             'has_data': False,
                             'inference_params': {'num_class': self.num_class,
                                                  'classes': self.classes},
                             'optimized': True,
                             'optimizer_info': {}
                             }
            shutil.copy2(self.temp_path + 'onnx_' + self.network_head + '_head_model.onnx',
                         head_metadata['model_paths'])
        with open(os.path.join(path, 'head_' + self.network_head + '.json'), 'w', encoding='utf-8') as f:
            json.dump(head_metadata, f, ensure_ascii=False, indent=4)

    def load(self, path=None):
        """
        Load implementation is meant for external usage to load a previously saved model for inference.
        :param path: the path of the model to be loaded
        :type path: str
        """
        if self.mode in ['backbone_only', 'head_only', 'finetune']:
            print("Loading backbone '{}'".format(self.backbone))
            self.__load_backbone(path)
        elif self.network_head == 'classifier' and self.mode == 'full':
            print("Loading backbone '{}' and head '{}'".format(self.backbone, self.network_head))
            self.__load_backbone(path)
            self.__load_head(path)

    def __load_backbone(self, path):
        if os.path.exists(os.path.join(path, 'backbone_' + self.backbone + '.json')):
            with open(os.path.join(path, 'backbone_' + self.backbone + '.json')) as f:
                metadata = json.load(f)
            self.threshold = metadata['inference_params']['threshold']
        else:
            raise UserWarning('No backbone_' + self.backbone + '.json found. Please have a check')
        if metadata['optimized']:
            if os.path.exists(os.path.join(path, 'onnx_' + self.backbone + '_backbone_model.onnx')):
                self.__load_from_onnx(path)
            else:
                raise UserWarning('No onnx_' + self.backbone + '_backbone_model.onnx found. Please have a check')
        else:
            if os.path.exists(os.path.join(path, 'backbone_' + self.backbone + '.pth')):
                self.__create_model(num_class=0)
                self.backbone_model.load_state_dict(torch.load(
                    os.path.join(path, 'backbone_' + self.backbone + '.pth'), map_location=torch.device(self.device)))
                self._model = {self.backbone_model, self.network_head_model}
            else:
                raise UserWarning('No backbone_' + self.backbone + '.pth found. Please have a check')

    def __load_head(self, path):
        if os.path.exists(os.path.join(path, 'head_' + self.network_head + '.json')):
            with open(os.path.join(path, 'head_' + self.network_head + '.json')) as f:
                metadata = json.load(f)
            self.classes = metadata['inference_params']['classes']
            self.num_class = metadata['inference_params']['num_class']
        else:
            raise UserWarning('No head_' + self.network_head + '.json found. Please have a check')
        if metadata['optimized']:
            if os.path.exists(os.path.join(path, 'onnx_' + self.network_head + '_head_model.onnx')):
                self.__load_from_onnx(path)
            else:
                raise UserWarning('No onnx_' + self.backbone + '_head_model.onnx found. Please have a check')
        else:
            if os.path.exists(os.path.join(path, 'head_' + self.network_head + '.pth')):
                self.__create_model(num_class=self.num_class)
                self._model = {self.backbone_model, self.network_head_model}
                self.network_head_model.load_state_dict(torch.load(
                    os.path.join(path, 'head_' + self.network_head + '.pth')))
            else:
                raise UserWarning('No head_' + self.network_head + '.pth found. Please have a check')

    def __load_from_onnx(self, path):
        path_backbone = os.path.join(path, 'onnx_' + self.backbone + '_backbone_model.onnx')
        self.ort_backbone_session = ort.InferenceSession(path_backbone)
        if self.mode == 'full' and self.network_head == 'classifier':
            path_head = os.path.join(path, 'onnx_' + self.network_head + '_head_model.onnx')
            self.ort_head_session = ort.InferenceSession(path_head)

    def __convert_to_onnx(self, verbose=False):
        if self.device == 'cuda':
            inp = torch.randn(1, 3, self.input_size[0], self.input_size[1]).cuda()
        else:
            inp = torch.randn(1, 3, self.input_size[0], self.input_size[1])
        input_names = ['data']
        output_names = ['features']
        output_name = os.path.join(self.temp_path, 'onnx_' + self.backbone + '_backbone_model.onnx')
        torch.onnx.export(self.backbone_model, inp, output_name, verbose=verbose, enable_onnx_checker=True,
                          input_names=input_names, output_names=output_names)
        if self.mode == 'full' and self.network_head == 'classifier':
            if self.device == 'cuda':
                inp = torch.randn(1, self.embedding_size).cuda()
            else:
                inp = torch.randn(1, self.embedding_size)
            input_names = ['features']
            output_names = ['classes']
            output_name = os.path.join(self.temp_path, 'onnx_' + self.network_head + '_head_model.onnx')
            torch.onnx.export(self.network_head_model, inp, output_name, verbose=verbose, enable_onnx_checker=True,
                              input_names=input_names, output_names=output_names)

    def optimize(self, do_constant_folding=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.__convert_to_onnx()
        self.__load_from_onnx(self.temp_path)

    def reset(self):
        pass
