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


from __future__ import print_function
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import json
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.data import SkeletonSequence
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR skeleton_based_action_recognition imports
from opendr.perception.skeleton_based_action_recognition.algorithm.models.stgcn import STGCN
from opendr.perception.skeleton_based_action_recognition.algorithm.models.tagcn import TAGCN
from opendr.perception.skeleton_based_action_recognition.algorithm.models.stbln import STBLN
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.feeder import Feeder
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.ntu_gendata import NTU60_CLASSES
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.kinetics_gendata import KINETICS400_CLASSES


class SpatioTemporalGCNLearner(Learner):
    def __init__(self, lr=1e-1, batch_size=128, optimizer_name='sgd', lr_schedule='',
                 checkpoint_after_iter=0, checkpoint_load_iter=0, temp_path='temp',
                 device='cuda', num_workers=32, epochs=50, experiment_name='stgcn_nturgbd',
                 device_ind=[0], val_batch_size=256, drop_after_epoch=[30, 40],
                 start_epoch=0, dataset_name='nturgbd_cv', num_class=60, num_point=25, num_person=2, in_channels=3,
                 graph_type='ntu', method_name='stgcn', stbln_symmetric=False, num_frames=300, num_subframes=100):
        super(SpatioTemporalGCNLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                       checkpoint_after_iter=checkpoint_after_iter,
                                                       checkpoint_load_iter=checkpoint_load_iter,
                                                       temp_path=temp_path, device=device)
        self.device = device
        self.device_ind = device_ind
        self.parent_dir = temp_path
        self.epochs = epochs
        self.num_workers = num_workers
        self.lr = lr
        self.base_lr = lr
        self.drop_after_epoch = drop_after_epoch
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.optimizer_name = optimizer_name
        self.experiment_name = experiment_name
        self.checkpoint_after_iter = checkpoint_after_iter
        self.checkpoint_load_iter = checkpoint_load_iter
        self.model_train_state = True
        self.ort_session = None
        self.dataset_name = dataset_name
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.graph_type = graph_type
        self.global_step = 0
        self.logging = False
        self.best_acc = 0
        self.start_epoch = start_epoch
        self.method_name = method_name
        self.stbln_symmetric = stbln_symmetric
        self.num_frames = num_frames
        self.num_subframes = num_subframes

        if self.num_subframes > self.num_frames:
            raise ValueError('number of subframes should be smaller than number of frames.')
        if self.dataset_name is None:
            raise ValueError(self.dataset_name +
                             "is not a valid dataset name. Supported datasets: nturgbd_cv, nturgbd_cs, kinetics")
        if self.graph_type is None:
            raise ValueError(self.graph_type +
                             "is not a valid graph type. Supported graphs: ntu, openpose")
        if self.method_name is None or self.method_name not in ['stgcn', 'tagcn', 'stbln']:
            raise ValueError(self.method_name +
                             "is not a valid dataset name. Supported methods: stgcn, tagcn, stbln")

        if self.device == 'cuda':
            self.output_device = self.device_ind[0] if type(self.device_ind) is list else self.device_ind
        self.__init_seed(1)

        if self.dataset_name in ['nturgbd_cv', 'nturgbd_cs']:
            self.classes_dict = NTU60_CLASSES
        elif self.dataset_name == 'kinetics':
            self.classes_dict = KINETICS400_CLASSES

    def fit(self, dataset, val_dataset, logging_path='', silent=False, verbose=True,
            momentum=0.9, nesterov=True, weight_decay=0.0001, train_data_filename='train_joints.npy',
            train_labels_filename='train_labels.pkl', val_data_filename="val_joints.npy",
            val_labels_filename="val_labels.pkl", skeleton_data_type='joint'):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.
        :param dataset: object that holds the training dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param val_dataset: object that holds the validation dataset
        :type val_dataset: ExternalDataset class object or DatasetIterator class object
        :param logging_path: path to save tensorboard log files. If set to None or '', tensorboard logging is
            disabled, defaults to ''
        :type logging_path: str, optional
        :param silent: if set to True, disables all printing of training progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param momentum: momentum value which is set in the optimizer
        :type momentum: float, optional
        :param nesterov: nesterov value which is set in the optimizer
        :type nesterov: bool, optional
        :param weight_decay: weight_decay value which is set in the optimizer
        :type weight_decay: float, optional
        :param train_data_filename: the file name of training data which is placed in the dataset path.
        :type train_data_filename: str, optional
        :param train_labels_filename: the file name of training labels which is placed in the dataset path.
        :type train_labels_filename: str, optional
        :param val_data_filename: the file name of val data which is placed in the dataset path.
        :type val_data_filename: str, optional
        :param val_labels_filename: the file name of val labels which is placed in the dataset path.
        :type val_labels_filename: str, optional
        :return: returns stats regarding the last evaluation ran
        :rtype: dict
        """
        self.logging_path = logging_path
        self.global_step = 0
        self.best_acc = 0
        # Tensorboard logging
        if self.logging_path != '' and self.logging_path is not None:
            self.logging = True
            self.tensorboard_logging_path = os.path.join(self.logging_path, self.experiment_name + '_tensorboard')
            if self.model_train_state:
                self.train_writer = SummaryWriter(os.path.join(self.tensorboard_logging_path, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(self.tensorboard_logging_path, 'val'), 'val')
            else:
                self.val_writer = SummaryWriter(os.path.join(self.tensorboard_logging_path, 'test'), 'test')
        else:
            self.logging = False
        # Initialize the model
        if self.model is None:
            self.init_model()
            if self.device == 'cuda':
                self.model = self.model.cuda(self.output_device)
                if type(self.device_ind) is list:
                    if len(self.device_ind) > 1:
                        self.model = nn.DataParallel(self.model, device_ids=self.device_ind,
                                                     output_device=self.output_device)
                self.loss = self.loss.cuda(self.output_device)
        # Load the model from a checkpoint
        checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
        if self.checkpoint_after_iter != 0 and not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)  # Checkpoints folder was just created
        if self.checkpoint_load_iter != 0:
            checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
            checkpoint_name = self.experiment_name + '-' + str(self.epochs - 1) + '-' + str(int(
                self.checkpoint_load_iter)) + '.pt'
            checkpoint_path = os.path.join(checkpoints_folder, checkpoint_name)
            self.__load_from_pt(checkpoint_path)
        if verbose:
            print("Model trainable parameters:", self.__count_parameters())
        # set the optimizer
        if self.optimizer_name == 'sgd':
            self.optimizer_ = optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay)
        elif self.optimizer_name == 'adam':
            self.optimizer_ = optim.Adam(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=weight_decay)
        else:
            raise ValueError(self.optimizer_ + "is not a valid optimizer name. Supported optimizers: sgd, adam")

        if self.lr_schedule != '':
            scheduler = self.lr_schedule
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_, milestones=self.drop_after_epoch, gamma=0.1,
                                                       last_epoch=-1, verbose=True)
        # load data
        traindata = self.__prepare_dataset(dataset,
                                           data_filename=train_data_filename,
                                           labels_filename=train_labels_filename,
                                           skeleton_data_type=skeleton_data_type,
                                           phase='train',
                                           verbose=verbose and not silent)

        train_loader = DataLoader(dataset=traindata,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  worker_init_fn=self.__init_seed(1))

        valdata = self.__prepare_dataset(val_dataset,
                                         data_filename=val_data_filename,
                                         labels_filename=val_labels_filename,
                                         skeleton_data_type=skeleton_data_type,
                                         phase='val',
                                         verbose=verbose and not silent)
        val_loader = DataLoader(dataset=valdata,
                                batch_size=self.val_batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                worker_init_fn=self.__init_seed(1))

        # start training
        self.global_step = self.start_epoch * len(train_loader) / self.batch_size
        # self.checkpoint_after_iter = int(len(train_loader) / self.batch_size)
        eval_results_list = []
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.__print_log('Training epoch: {}'.format(epoch + 1))
            save_model = ((self.global_step + 1) % self.checkpoint_after_iter == 0) or (epoch + 1 == self.epochs)
            # save_model = ((epoch + 1) % self.checkpoint_after_iter == 0) or (epoch + 1 == self.epochs)
            loss_value = []
            if self.logging:
                self.train_writer.add_scalar('epoch', epoch, self.global_step)
            self.__record_time()
            timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
            process = tqdm(train_loader)
            for batch_idx, (data, label, index) in enumerate(process):
                self.global_step += 1
                # get data
                if self.device == 'cuda':
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                else:
                    data = Variable(data.float(), requires_grad=False)
                    label = Variable(label.long(), requires_grad=False)
                timer['dataloader'] += self.__split_time()

                # forward
                output = self.model(data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label) + l1

                # backward
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                loss_value.append(loss.data.item())
                timer['model'] += self.__split_time()

                value, predict_label = torch.max(output.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                if self.logging:
                    self.train_writer.add_scalar('acc', acc, self.global_step)
                    self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                    self.train_writer.add_scalar('loss_l1', l1, self.global_step)

                # statistics
                self.lr = self.optimizer_.param_groups[0]['lr']
                if self.logging:
                    self.train_writer.add_scalar('lr', self.lr, self.global_step)
                timer['statistics'] += self.__split_time()

            # statistics of time consumption and loss
            proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                          for k, v in timer.items()}
            self.__print_log('\t Mean training loss: {:.4f}.'.format(np.mean(loss_value)))
            self.__print_log('\t Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
            if save_model:
                checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
                checkpoint_name = self.experiment_name + '-' + str(epoch) + '-' + str(int(self.global_step))
                self.ort_session = None
                self.save(path=checkpoints_folder, model_name=checkpoint_name)
            eval_results = self.eval(val_dataset, val_loader=val_loader, epoch=epoch,
                                     val_data_filename=val_data_filename, val_labels_filename=val_labels_filename)
            eval_results_list.append(eval_results)
            scheduler.step()
        if verbose:
            print('best accuracy: ', self.best_acc, ' model_name: ', self.experiment_name)
        return {"train_loss": np.mean(loss_value), "eval_results": eval_results_list,
                "best_accuracy": self.best_acc, "model_name": self.experiment_name}

    def eval(self, val_dataset, val_loader=None, epoch=0, silent=False, verbose=True,
             val_data_filename='val_joints.npy', val_labels_filename='val_labels.pkl', skeleton_data_type='joint',
             save_score=False, wrong_file=None, result_file=None, show_topk=[1, 5]):
        """
        This method is used for evaluating the algorithm on a val dataset.
        :param val_dataset: object that holds the val dataset
        :type val_dataset: ExternalDataset class object or DatasetIterator class object
        :param epoch: the number of epochs that the method is trained up to now. Default to 0 when we validate a
        pretrained model.
        :type epoch: int, optional
        :param silent: if set to True, disables all printing of training progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param val_data_filename: the file name of val data which is placed in the dataset path.
        :type val_data_filename: str, optional
        :param val_labels_filename: the file name of val labels which is placed in the dataset path.
        :type val_labels_filename: str, optional
        :param skeleton_data_type: the data stream that should be used for training and evaluation. Defalut is set to
        'joint'
        :type skeleton_data_type: str
        :param save_score: if set to True, it saves the classification score of all samples in differenc classes
        in a log file. Default to False.
        :type save_score: bool, optional
        :param wrong_file: if set to True, it saves the results of wrongly classified samples. Default to False.
        :type wrong_file: bool, optional
        :param result_file: if set to True, it saves the classification results of all samples. Default to False.
        :type result_file: bool, optional
        :param show_topk: is set to a list of integer numbers defining the k in top-k accuracy. Default is set to [1,5].
        :type show_topk: list, optional
        :return: returns stats regarding the last evaluation ran
        :rtype: dict
        """

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        # load data
        if val_loader is None:
            valdata = self.__prepare_dataset(val_dataset,
                                             data_filename=val_data_filename,
                                             labels_filename=val_labels_filename,
                                             skeleton_data_type=skeleton_data_type,
                                             phase='val',
                                             verbose=verbose and not silent)
            val_loader = DataLoader(dataset=valdata,
                                    batch_size=self.val_batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    drop_last=False,
                                    worker_init_fn=self.__init_seed(1))
        self.model.eval()
        self.__print_log('Eval epoch: {}'.format(epoch + 1))
        loss_value = []
        score_frag = []
        process = tqdm(val_loader)
        for batch_idx, (data, label, index) in enumerate(process):
            with torch.no_grad():
                if self.device == "cuda":
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                else:
                    data = Variable(data.float(), requires_grad=False)
                    label = Variable(label.long(), requires_grad=False)
                output = self.model(data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.item())
                _, predict_label = torch.max(output.data, 1)

            if wrong_file is not None or result_file is not None:
                predict = list(predict_label.cpu().numpy())
                true = list(label.data.cpu().numpy())
                for i, x in enumerate(predict):
                    if result_file is not None:
                        f_r.write(str(x) + ',' + str(true[i]) + '\n')
                    if x != true[i] and wrong_file is not None:
                        f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        score = np.concatenate(score_frag)
        loss = np.mean(loss_value)
        accuracy = val_loader.dataset.top_k(score, 1)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
        if verbose:
            print('Accuracy: ', accuracy, ' model: ', self.experiment_name)
        if self.model_train_state and self.logging:
            self.val_writer.add_scalar('loss', loss, self.global_step)
            self.val_writer.add_scalar('loss_l1', l1, self.global_step)
            self.val_writer.add_scalar('acc', accuracy, self.global_step)

        score_dict = dict(zip(val_loader.dataset.sample_name, score))
        self.__print_log('\tMean {} loss of {} batches: {}.'.format(
            'val', len(val_loader), np.mean(loss_value)))
        for k in show_topk:
            self.__print_log('\tTop{}: {:.2f}%'.format(
                k, 100 * val_loader.dataset.top_k(score, k)))
        if save_score and self.logging:
            with open('{}/epoch{}_{}_score.pkl'.format(self.logging_path, epoch + 1, 'val'), 'wb') as f:
                pickle.dump(score_dict, f)
        return {"epoch": epoch, "accuracy": accuracy, "loss": loss, "score": score}

    def __prepare_dataset(self, dataset, data_filename="train_joints.npy",
                          labels_filename="train_labels.pkl",
                          skeleton_data_type='joint',
                          phase='train',
                          verbose=True):
        """
        This internal method prepares the train dataset depending on what type of dataset is provided.
        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation.
        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.
        :param dataset: the dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object

        :param data_filename: the data file name which is placed in the dataset path.
        :type data_filename: str, optional
        :param labels_filename: the label file name which is placed in the dataset path.
        :type labels_filename: str, optional
        :param verbose: whether to print additional information, defaults to 'True'
        :type verbose: bool, optional
        :raises UserWarning: UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames
        :return: returns Feeder class object or DatasetIterator class object
        :rtype: Feeder class object or DatasetIterator class object
        """
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() != "nturgbd" and dataset.dataset_type.lower() != "kinetics":
                raise UserWarning("dataset_type must be \"NTURGBD or Kinetics\"")
            # Get data and labels path
            data_path = os.path.join(dataset.path, data_filename)
            labels_path = os.path.join(dataset.path, labels_filename)
            if phase == 'train':
                if dataset.dataset_type.lower() == "nturgbd" or self.method_name == 'tagcn':
                    random_choose = False
                    random_move = False
                    window_size = -1
                elif dataset.dataset_type.lower() == "kinetics":
                    random_choose = True
                    random_move = True
                    window_size = 150
            else:
                random_choose = False
                random_move = False
                window_size = -1

            if verbose:
                print('Dataset path is set. Loading feeder...')
            return Feeder(data_path=data_path, label_path=labels_path, random_choose=random_choose,
                          random_move=random_move, window_size=window_size,
                          skeleton_data_type=skeleton_data_type,
                          data_name=dataset.dataset_type.lower())
        elif isinstance(dataset, DatasetIterator):
            return dataset

    def init_model(self):
        """Initializes the imported model."""
        cuda_ = (self.device == 'cuda')
        if self.method_name == 'stgcn':
            self.model = STGCN(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person,
                               in_channels=self.in_channels, graph_type=self.graph_type,
                               cuda_=cuda_)
            if self.logging:
                shutil.copy2(inspect.getfile(STGCN), self.logging_path)
        elif self.method_name == 'tagcn':
            self.model = TAGCN(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person,
                               in_channels=self.in_channels, graph_type=self.graph_type,
                               num_frames=self.num_frames, num_selected_frames=self.num_subframes, cuda_=cuda_)
            if self.logging:
                shutil.copy2(inspect.getfile(TAGCN), self.logging_path)
        elif self.method_name == 'stbln':
            self.model = STBLN(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person,
                               in_channels=self.in_channels, symmetric=self.stbln_symmetric,
                               cuda_=cuda_)
            if self.logging:
                shutil.copy2(inspect.getfile(STBLN), self.logging_path)
        self.loss = nn.CrossEntropyLoss()
        # print(self.model)

    def infer(self, SkeletonSeq_batch):
        """
        This method performs inference on the batch provided.
        :param skeletonseq_batch: Object that holds a batch of data to run inference on.
        The data is a sequence of skeletons (of an action video).
        :type skeletonseq_batch: Data class type
        :return: A list of predicted targets.
        :rtype: list of Target class type objects.
        """

        if not isinstance(SkeletonSeq_batch, SkeletonSequence):
            SkeletonSeq_batch = SkeletonSequence(SkeletonSeq_batch)
        SkeletonSeq_batch = torch.from_numpy(SkeletonSeq_batch.numpy())

        if self.device == "cuda":
            SkeletonSeq_batch = Variable(SkeletonSeq_batch.float().cuda(self.output_device), requires_grad=False)
        else:
            SkeletonSeq_batch = Variable(SkeletonSeq_batch.float(), requires_grad=False)
        if self.ort_session is not None:
            output = self.ort_session.run(None, {'data': np.array(SkeletonSeq_batch.cpu())})
        else:
            if self.model is None:
                raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
            if self.model_train_state:
                self.model.eval()
                self.model_train_state = False
            with torch.no_grad():
                output = self.model(SkeletonSeq_batch)
        if isinstance(output, tuple):
            output, l1 = output
        else:
            output = output

        m = nn.Softmax(dim=0)
        softmax_predictions = m(output.data[0])
        class_confidence = float(torch.max(softmax_predictions))
        class_ind = int(torch.argmax(softmax_predictions))
        class_description = self.classes_dict[class_ind]
        category = Category(prediction=class_ind, confidence=class_confidence, description=class_description)

        return category

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
            self.__convert_to_onnx(os.path.join(self.parent_dir, self.experiment_name, "onnx_model.onnx"),
                                   do_constant_folding)
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(os.path.join(self.parent_dir, self.experiment_name), exist_ok=True)
            self.__convert_to_onnx(os.path.join(self.parent_dir, self.experiment_name, "onnx_model.onnx"),
                                   do_constant_folding, verbose=False)

        self.__load_from_onnx(os.path.join(self.parent_dir, self.experiment_name, "onnx_model.onnx"))

    def __convert_to_onnx(self, output_name, do_constant_folding=False, verbose=False):
        """
        Converts the loaded regular PyTorch model to an ONNX model and saves it to disk.
        :param output_name: path and name to save the model, e.g. "/models/onnx_model.onnx"
        :type output_name: str
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        # Input to the model
        c, t, v, m = [self.in_channels, 300, self.num_point, self.num_person]
        n = self.batch_size
        onnx_input = torch.randn(n, c, t, v, m)
        if self.device == "cuda":
            onnx_input = Variable(onnx_input.float().cuda(self.output_device), requires_grad=False)
        else:
            onnx_input = Variable(onnx_input.float(), requires_grad=False)
        # torch_out = self.model(onnx_input)
        # Export the model
        torch.onnx.export(self.model,  # model being run
                          onnx_input,  # model input (or a tuple for multiple inputs)
                          output_name,  # where to save the model (can be a file or file-like object)
                          verbose=verbose,
                          opset_version=11,
                          enable_onnx_checker=True,
                          do_constant_folding=do_constant_folding,
                          input_names=['onnx_input'],  # the model's input names
                          output_names=['onnx_output'],  # the model's output names
                          dynamic_axes={'onnx_input': {0: 'n'},  # variable lenght axes
                                        'onnx_output': {0: 'n'}})

    def save(self, path, model_name='', verbose=True):
        """
        This method is used to save a trained model.
        Provided with the path and model_name, it saves the model there with a proper format and a .json file
        with metadata. If self.optimize was ran previously, it saves the optimized ONNX model in a similar fashion,
        by copying it from the self.temp_path it was saved previously during conversion.
        :param path: for the model to be saved
        :type path: str
        :param model_name: the name of the file to be saved
        :type model_name: str
        :param epoch: if model_name is not provided, experiment_name, epoch and global_step are used to make the file
        name to show the epoch and global_step that the saved model belongs to.
        :type epoch: int
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        if self.model is None and self.ort_session is None:
            raise UserWarning("No model is loaded, cannot save.")
        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}}

        if not os.path.exists(path):
            os.makedirs(path)
        if self.ort_session is None:
            checkpoint_name = model_name + '.pt'
            checkpoint_path = os.path.join(path, checkpoint_name)
            model_metadata["model_paths"] = [checkpoint_path]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pt"
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, checkpoint_path)
            if verbose:
                print("Saved Pytorch model.")
        else:
            checkpoint_name = model_name + '.onnx'
            checkpoint_path = os.path.join(path, checkpoint_name)
            model_metadata["model_paths"] = [checkpoint_path]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"
            # Copy already optimized model from temp path
            shutil.copy2(os.path.join(self.parent_dir, self.experiment_name, "onnx_model.onnx"),
                         model_metadata["model_paths"][0])
            model_metadata["optimized"] = True
            if verbose:
                print("Saved ONNX model.")

        json_model_name = model_name + '.json'
        json_model_path = os.path.join(path, json_model_name)
        with open(json_model_path, 'w') as outfile:
            json.dump(model_metadata, outfile)

    def load(self, path, model_name, verbose=True):
        """
        Loads the model from inside the path provided, based on the metadata.json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param model_name: the name of saved_model
        :type model_name: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)
        if not metadata["optimized"]:
            self.__load_from_pt(os.path.join(path, model_name + '.pt'))
            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.__load_from_onnx(os.path.join(path, model_name + '.onnx'))
            if verbose:
                print("Loaded ONNX model.")

    def __load_from_pt(self, path, verbose=True):
        """Loads the .pt model weights (or checkpoint) from the path provided.
        :param path: path of the directory the model (checkpoint) was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'True'
        :type verbose: bool, optional
        """
        if path is not None:
            self.__print_log('Load weights from {}'.format(path))
            try:
                weights = torch.load(path)
            except FileNotFoundError as e:
                e.strerror = "Pretrained weights '.pt' file must be placed in path provided. \n " \
                             "No such file or directory."
                raise e
            if verbose:
                print("Loading checkpoint")
            if self.device == "cuda":
                weights = OrderedDict(
                    [[k.split('module.')[-1], v.cuda(self.output_device)] for k, v in weights.items()])
            else:
                weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])
                # keys = list(weights.keys())
            try:
                self.init_model()
                self.model.load_state_dict(weights)
            except Exception:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Could not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
            if self.device == "cuda":
                self.model = self.model.cuda(self.output_device)
                if type(self.device_ind) is list:
                    if len(self.device_ind) > 1:
                        self.model = nn.DataParallel(self.model, device_ids=self.device_ind,
                                                     output_device=self.output_device)
                self.loss = self.loss.cuda(self.output_device)

    def __load_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.
        :param path: path to ONNX model
        :type path: str
        """
        self.ort_session = onnxruntime.InferenceSession(path)

    def multi_stream_eval(self, dataset, scores, data_filename='val_joints.npy',
                          labels_filename='val_labels.pkl', skeleton_data_type='joint',
                          verbose=True, silent=True):
        """
        :param dataset: ExternalDataset class object.
        :type dataset: object that holds the validation dataset
        :param scores: a list of score arrays. Each array in the list contains the evaluation results for a dataset.
        :type scores: a list of score arrays.
        :param data_filename: the data file name which is placed in the dataset path.
        :type data_filename: str, optional
        :param labels_filename: the labels file name which is placed in the dataset path.
        :type labels_filename: str, optional
        :param skeleton_data_type: the data stream that should be used for training and evaluation. Defalut is set to
        'joint'.
        :type skeleton_data_type: str, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param silent: if set to True, disables all printing of training progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :return: the top_k classification results

        """
        valdata = self.__prepare_dataset(dataset,
                                         data_filename=data_filename,
                                         labels_filename=labels_filename,
                                         skeleton_data_type=skeleton_data_type,
                                         phase='val',
                                         verbose=verbose and not silent)
        val_loader = DataLoader(dataset=valdata,
                                batch_size=self.val_batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                worker_init_fn=self.__init_seed(1))
        total_score = 0
        for i in range(len(scores)):
            total_score += scores[i]
        accuracy_top1 = val_loader.dataset.top_k(total_score, 1)
        accuracy_top5 = val_loader.dataset.top_k(total_score, 1)

        if self.logging:
            self.__print_log('\tMulti_Stream_Top{}: {:.2f}%'.format(
                1, 100 * accuracy_top1))
            self.__print_log('\tMulti_Stream_Top{}: {:.2f}%'.format(
                5, 100 * accuracy_top5))
            with open('{}/multistream_score.pkl'.format(self.logging_path), 'wb') as f:
                pickle.dump(total_score, f)
        return total_score

    def download(self, path=None, method_name="stgcn", mode="pretrained", verbose=True,
                 url=OPENDR_SERVER_URL + "perception/skeleton_based_action_recognition/",
                 file_name='stgcn_nturgbd_cv_joint-49-29400'):
        """
        Download utility for various skeleton_based_action_recognition components. Downloads files depending on mode and
        saves them in the path provided. It supports downloading:
        - The pretrained models
        - Train, Val, Test datasets
        :param path: Local path to save the files, defaults to self.temp_path if None
        :type path: str, path, optional
        :param mode: What file to download, can be one of "pretrained", "train_data", "val_data", "test_data",
        defaults to "pretrained"
        :type mode: str, optional
        :param verbose: Whether to print messages in the console, defaults to False
        :type verbose: bool, optional
        :param url: URL of the FTP server, defaults to OpenDR FTP URL
        :type url: str, optional
        :param file_name: the name of the file containing the pretrained model.
        :type file_name: str
        """
        valid_modes = ["pretrained", "train_data", "val_data", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.parent_dir

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, self.dataset_name)):
            os.makedirs(os.path.join(path, self.dataset_name))
        if not os.path.exists(os.path.join(path, '{}_checkpoints'.format(self.experiment_name), file_name)):
            os.makedirs(os.path.join(path, '{}_checkpoints'.format(self.experiment_name), file_name))

        if mode == "pretrained":
            if verbose:
                print("Downloading pretrained model...")
            # download the .json model
            if not os.path.exists(os.path.join(path, '{}_checkpoints'.format(self.experiment_name),
                                               file_name+'.json')):
                file_url = os.path.join(url, 'pretrained_models', method_name, file_name, file_name+'.json')
                urlretrieve(file_url, os.path.join(path, '{}_checkpoints'.format(self.experiment_name), file_name,
                                                   file_name+'.json'))
                if verbose:
                    print("Downloaded metadata json.")
            else:
                if verbose:
                    print("Metadata json file already exists.")
            # download the .pt model
            if not os.path.exists(os.path.join(path, '{}_checkpoints'.format(self.experiment_name),
                                               file_name+'.pt')):
                file_url = os.path.join(url, 'pretrained_models', method_name, file_name, file_name+'.pt')
                urlretrieve(file_url, os.path.join(path, '{}_checkpoints'.format(self.experiment_name), file_name,
                                                   file_name+'.pt'))
            else:
                if verbose:
                    print("Trained model.pt file already exists.")
            if verbose:
                print("Pretrained model download complete.")
            downloaded_files_path = os.path.join(path, '{}_checkpoints'.format(self.experiment_name), file_name)

        elif mode == "train_data":
            if verbose:
                print("Downloading train data...")
            if not os.path.exists(os.path.join(path, self.dataset_name, "train_joints.npy")):
                # Download train data
                file_url = os.path.join(url, 'data', self.dataset_name, "train_joints.npy")
                urlretrieve(file_url,
                            os.path.join(path, self.dataset_name, "train_joints.npy"))
            else:
                if verbose:
                    print("train_data file already exists.")
            # Download labels
            if not os.path.exists(os.path.join(path, self.dataset_name, "train_labels.pkl")):
                file_url = os.path.join(url, 'data', self.dataset_name, "train_labels.pkl")
                urlretrieve(file_url,
                            os.path.join(path, self.dataset_name, "train_labels.pkl"))
            else:
                if verbose:
                    print("train_labels file already exists.")
            if verbose:
                print("Train data download complete.")
            downloaded_files_path = os.path.join(path, self.dataset_name)

        elif mode == "val_data":
            if verbose:
                print("Downloading validation data...")
            if not os.path.exists(os.path.join(path, self.dataset_name, "val_joints.npy")):
                # Download val data
                file_url = os.path.join(url, 'data', self.dataset_name, "val_joints.npy")
                urlretrieve(file_url,
                            os.path.join(path, self.dataset_name, "val_joints.npy"))
            else:
                if verbose:
                    print("val_data file already exists.")
            # Download labels
            if not os.path.exists(os.path.join(path, self.dataset_name, "val_labels.pkl")):
                file_url = os.path.join(url, 'data', self.dataset_name, "val_labels.pkl")
                urlretrieve(file_url,
                            os.path.join(path, self.dataset_name, "val_labels.pkl"))
            else:
                if verbose:
                    print("val_labels file already exists.")
            if verbose:
                print("Val data download complete.")
            downloaded_files_path = os.path.join(path, self.dataset_name)

        elif mode == "test_data":
            if verbose:
                print("Downloading test data...")
            if not os.path.exists(os.path.join(path, self.dataset_name, "val_joints.npy")):
                # Download test data
                file_url = os.path.join(url, 'data', self.dataset_name, "val_joints.npy")
                urlretrieve(file_url,
                            os.path.join(path, self.dataset_name, "val_joints.npy"))
            else:
                if verbose:
                    print("test_data file already exists.")
            if verbose:
                print("Test data download complete.")
            downloaded_files_path = os.path.join(path, self.dataset_name, "val_joints.npy")

        return downloaded_files_path

    def __record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def __split_time(self):
        split_time = time.time() - self.cur_time
        self.__record_time()
        return split_time

    def __print_log(self, str_log, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str_log = "[ " + localtime + ' ] ' + str_log
        if self.logging:
            with open('{}/log.txt'.format(self.logging_path), 'a') as f:
                print(str_log, file=f)

    def __count_parameters(self):
        """
        Returns the number of the model's trainable parameters.
        :return: number of trainable parameters
        :rtype: int
        """
        if self.model is None:
            raise UserWarning("Model is not initialized, can't count trainable parameters.")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init_seed(self, seed):
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError
