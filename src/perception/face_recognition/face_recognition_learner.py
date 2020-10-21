# Copyright 1996-2020 OpenDR European Project
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
import sys

from engine.learners import Learner
from engine.data import Image
from perception.face_recognition.algorithm.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from perception.face_recognition.algorithm.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from perception.face_recognition.algorithm.backbone.model_mobilenet import MobileFaceNet
from perception.face_recognition.algorithm.head.losses import ArcFace, CosFace, SphereFace, Am_softmax, Classifier
from perception.face_recognition.algorithm.loss.focal import FocalLoss
from perception.face_recognition.algorithm.util.utils import make_weights_for_balanced_classes, get_val_data, \
    separate_irse_bn_paras, separate_mobilenet_bn_paras, l2_norm, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, buffer_val, AverageMeter, accuracy


class FaceRecognition(Learner):
    def __init__(self, lr=0.1, iters=120, batch_size=32, optimizer='sgd', device='cuda', threshold=0.0,
                 backbone='ir_50', network_head='arcface', loss='focal',
                 temp_path='./temp', mode='backbone_only',
                 checkpoint_after_iter=10, checkpoint_load_iter=0, val_after=10,
                 input_size=None, rgb_mean=None, rgb_std=None, embedding_size=512,
                 weight_decay=5e-4, momentum=0.9, drop_last=True, stages=None,
                 pin_memory=True, num_workers=4,
                 seed=123):
        super(FaceRecognition, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
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
            stages = [8, 16, 24]
        if self.device == 'cuda':
            gpu_id = [0]
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
        self.ort_session = None  # ONNX runtime inference session

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
                             'mobilefacenet': MobileFaceNet(self.input_size)}
            backbone = backbone_dict[self.backbone]
            self.backbone_model = backbone.to(self.device)
        # Create the head architecture
        if self.mode != 'backbone_only':
            head_dict = {
                'arcface': ArcFace(in_features=self.embedding_size, out_features=self.num_class, device_id=self.gpu_id),
                'cosface': CosFace(in_features=self.embedding_size, out_features=self.num_class, device_id=self.gpu_id),
                'sphereface': SphereFace(in_features=self.embedding_size, out_features=self.num_class,
                                         device_id=self.gpu_id),
                'am_softmax': Am_softmax(in_features=self.embedding_size, out_features=self.num_class,
                                         device_id=self.gpu_id),
                'classifier': Classifier(in_features=self.embedding_size, out_features=self.num_class,
                                         device=self.device)}
            head = head_dict[self.network_head]
            self.network_head_model = head.to(self.device)
        else:
            self.network_head_model = None

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

        eval_results = {}
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

        if dataset.type != 'imagefolder':
            sys.exit('dataset should be of type imagefolder')

        dataset_train = datasets.ImageFolder(dataset.path, train_transform)

        # create a weighted random sampler to process imbalanced data
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
                head_info = torch.load(os.path.join(self.temp_path, 'checkpoint', 'head_{}_iter_{}'.format(
                    self.network_head, self.epoch)))
                self.network_head_model.load_state_dict(head_info['head_state_dict'])
            else:
                backbone_info = torch.load(os.path.join(self.temp_path, 'checkpoint', 'backbone_{}_iter_{}'.format(
                    self.backbone, self.epoch)))
                self.backbone_model.load_state_dict(backbone_info['backbone_state_dict'])
                head_info = torch.load(os.path.join(self.temp_path, 'checkpoint', 'head_{}_iter_{}'.format(
                    self.network_head, self.epoch)))
                self.network_head_model.load_state_dict(head_info['head_state_dict'])

        # separate batch_norm parameters from others
        # do not do weight decay for batch_norm parameters to improve the generalizability
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

        # ======= train & validation & save checkpoint =======#
        disp_freq = len(train_loader) // 100  # frequency to display training loss & acc
        if disp_freq < 1:
            disp_freq = 1000

        num_epoch_warm_up = self.iters // 25  # use the first 1/25 epochs to warm up
        num_batch_warm_up = len(train_loader) * num_epoch_warm_up  # use the first 1/25 epochs to warm up

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
        for epoch in range(self.iters):  # start training process
            if self.epoch == self.stages[0]:  # adjust LR for each training stage after warm up
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
                        batch + 1 <= num_batch_warm_up):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, num_batch_warm_up, self.lr, optimizer)

                # compute output
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                features = self.backbone_model(inputs)
                outputs = self.network_head_model(features, labels)
                loss = criterion(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # display training loss & acc every disp_freq
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

                batch += 1  # batch index

            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
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
                    eval_results = self.eval(val_dataset)
            if self.checkpoint_after_iter != 0:
                if self.epoch % self.checkpoint_after_iter == 0:
                    self.__save(os.path.join(self.temp_path, 'checkpoint'))

        return eval_results

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
        if self._model is None and self.ort_session is None:
            sys.exit('A model should be loaded first')
        if os.path.exists(os.path.join(save_path, 'reference.pkl')):
            print('Loading Reference')
            self.database = pickle.load(open(os.path.join(save_path, 'reference.pkl'), "rb"))
        else:
            database = {}
            transform = transforms.Compose([
                transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
                transforms.CenterCrop([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)]
            )
            with torch.no_grad():
                for subdir, dirs, files in os.walk(path):
                    total = 0
                    features_sum = torch.zeros(1, self.embedding_size).to(self.device)
                    for file in files:
                        total += 1
                        inputs = cv2.imread(os.path.join(subdir, file))
                        inputs = transform(PILImage.fromarray(
                            cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB))
                        )
                        inputs = inputs.unsqueeze(0)
                        if self.ort_session is not None:
                            features = self.ort_session.run(None, {'data': np.array(inputs.cpu())})
                            features = torch.tensor(features[0])
                        else:
                            self.backbone_model.eval()
                            inputs = inputs.to(self.device)
                            features = self.backbone_model(inputs)
                        features = l2_norm(features)
                        features_sum += features
                    avg_features = features_sum / total
                    if subdir not in database:
                        database[subdir] = [avg_features]
                    else:
                        database[subdir].append(avg_features)
            f = open(os.path.join(save_path, "reference.pkl"), "wb")
            pickle.dump(database, f)
            f.close()
            self.database = database

    def infer(self, img=None):
        if not isinstance(img, Image):
            img = Image(img)
        img = img.numpy()
        if self._model is None and self.ort_session is None:
            sys.exit('A model should be loaded first')
        transform = transforms.Compose([
            transforms.Resize([int(128 * self.input_size[0] / 112), int(128 * self.input_size[0] / 112)]),
            transforms.CenterCrop([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)]
        )
        if self.mode == 'backbone_only':
            distance = self.threshold
            if distance == 0:
                distance = 1000
            to_keep = None
            if self.database is None:
                sys.exit('A reference for comparison should be created first. Try calling fit_reference()')
            self.backbone_model.eval()
            with torch.no_grad():
                img = PILImage.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = transform(img)
                img = img.unsqueeze(0)
                img = img.to(self.device)
                if self.ort_session is not None:
                    features = self.ort_session.run(None, {'data': np.array(img.cpu())})
                    features = torch.tensor(features[0])
                else:
                    self.backbone_model.eval()
                    features = self.backbone_model(img)
                features = l2_norm(features)
            for key in self.database:
                for item in self.database[key]:
                    diff = np.subtract(features.cpu().numpy(), item.cpu().numpy())
                    dist = np.sum(np.square(diff), axis=1)
                    if np.isnan(dist):
                        dist = 1000
                    if dist < distance:
                        distance = dist
                        to_keep = key
                        to_keep = to_keep.split('/')
                        to_keep = to_keep[-1]
            return to_keep

        elif self.network_head == 'classifier':
            self.backbone_model.eval()
            self.network_head_model.eval()
            with torch.no_grad():
                img = transform(PILImage.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                )
                img = img.unsqueeze(0)
                img = img.to(self.device)
                features = self.backbone_model(img)
                features = l2_norm(features)
                outs = self.network_head_model(features)
                _, predicted = torch.max(outs.data, 1)
            return self.classes[predicted.item()]
        else:
            raise NameError('Infer should be called either with backbone_only mode or with a classifier head')

    def eval(self, dataset=None):
        if self._model is None:
            sys.exit('A model should be loaded first')
        if self.network_head != 'classifier' and self.mode != 'head_only':
            self.backbone_model.eval()
            if self.data is None or self.pairs is None:
                self.data, self.pairs = get_val_data(dataset.path, dataset.dataset_type)
            if True:
                print("=" * 60)
                print("Perform Evaluation on " + dataset.dataset_type)
            eval_accuracy, best_threshold, roc_curve = perform_val(False, self.device, self.embedding_size,
                                                                   self.batch_size, self.backbone_model, self.data,
                                                                   self.pairs)

            self.threshold = float(best_threshold)
            if self.logging:
                buffer_val(self.writer, dataset.dataset_type, eval_accuracy, best_threshold, roc_curve, self.epoch + 1)
            if True:
                print(
                    "Evaluation on " + dataset.dataset_type + ": Acc: {} ".format(eval_accuracy))
                print("=" * 60)
            return {'Accuracy': eval_accuracy, 'Best_Threshold': best_threshold}

        else:
            self.backbone_model.eval()
            if self.mode != 'backbone_only':
                self.network_head_model.eval()
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

            print('Accuracy of the network on {} test images: {}'.format(total,
                                                                         100 * correct / total))
            return {"Accuracy": 100 * correct / total}

    def __save(self, path=None):
        """
        Internal save implementation is used to create checkpoints. Provided with a path,
        it adds training state information in a custom dict.
        :param path: path for the model to be saved
        :type path: str
        """
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

    def save(self, path=None):
        """
        Save for external usage.
        This will be loaded with self.load.
        :param path for the model to be saved
        :type path: str
        """
        backbone_custom_dict = {'backbone_state_dict': self.backbone_model.state_dict(),
                                'threshold': self.threshold}
        head_custom_dict = {'head_state_dict': self.network_head_model.state_dict(),
                            'classes': self.classes,
                            'num_class': self.num_class}
        if self.mode == 'head_only':
            torch.save(head_custom_dict, os.path.join(path, 'head_{}'.format(self.network_head)))
        else:
            torch.save(backbone_custom_dict, os.path.join(path, 'backbone_{}'.format(self.backbone)))
            torch.save(head_custom_dict, os.path.join(path, 'head_{}'.format(self.network_head)))

    def load(self, path=None):
        """
        Load implementation is meant for external usage to load a previously saved model for inference.
        :param path: the path of the model to be loaded
        :type path: str
        """
        if self.mode == 'backbone_only' or self.mode == 'head_only':
            if os.path.isfile(os.path.join(path, 'backbone_{}'.format(self.backbone))):
                backbone_info = torch.load(os.path.join(path, 'backbone_{}'.format(self.backbone)))
                print("Loading backbone '{}'".format(self.backbone))
                self.__create_model(num_class=0)
                self._model = {self.backbone_model, self.network_head_model}
                self.backbone_model.load_state_dict(backbone_info['backbone_state_dict'])
                self.threshold = backbone_info['threshold']
            else:
                sys.exit("No file Backbone_{} found in '{}'. Please Have a Check".format(self.backbone, path))
        elif self.network_head == 'classifier':
            if os.path.isfile(os.path.join(path, 'head_{}'.format(self.network_head))) and os.path.isfile(
                    os.path.join(path, 'backbone_{}'.format(self.backbone))):
                print("Loading backbone '{}' and head '{}'".format(self.backbone, self.network_head))
                head_info = torch.load(os.path.join(path, 'head_{}'.format(self.network_head)))
                backbone_info = torch.load(os.path.join(path, 'backbone_{}'.format(self.backbone)))
                self.__create_model(num_class=head_info['num_class'])
                self._model = {self.backbone_model, self.network_head_model}
                self.network_head_model.load_state_dict(head_info['head_state_dict'])
                self.classes = head_info['classes']
                self.backbone_model.load_state_dict(backbone_info['backbone_state_dict'])
                self.threshold = backbone_info['threshold']
            else:
                sys.exit(
                    "No file head_{} or backbone_{} found in '{}'. Please have a check".format(
                        self.network_head, self.backbone, path))

    def load_from_onnx(self, path):
        self.ort_session = ort.InferenceSession(path)

    def convert_to_onnx(self, output_name, do_constant_folding=False):
        inp = torch.randn(1, 3, 112, 112).cuda()
        input_names = ['data']
        output_names = ['features']

        torch.onnx.export(self.backbone_model, inp, output_name, verbose=True, do_constant_folding=do_constant_folding,
                          input_names=input_names, output_names=output_names)

    def optimize(self, path, do_constant_folding=False):
        """
        Optimize method saves the model in onnx format in the path specified.
        The saved model can then be used with load_from_onnx() method.
        """
        self.convert_to_onnx(path, do_constant_folding)

    def reset(self):
        pass
