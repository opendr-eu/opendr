# MIT License
#
# Copyright (c) 2021 Dat Thanh Tran
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
import torch.optim as optim
import pickle
from tqdm import tqdm
import os
import time
import numpy as np


def get_cosine_lr_scheduler(init_lr, final_lr):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = final_lr + 0.5 * (init_lr - final_lr) * (1 + np.cos(np.pi * epoch_idx / n_epoch))
        return lr

    return lr_scheduler


def get_multiplicative_lr_scheduler(init_lr, drop_at, multiplicative_factor):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = init_lr
        for epoch in drop_at:
            if epoch_idx + 1 >= epoch:
                lr *= multiplicative_factor
        return lr

    return lr_scheduler


class ClassifierTrainer:
    n_test_minibatch = 10

    def __init__(self,
                 n_epoch,
                 epoch_idx,
                 lr_scheduler,
                 optimizer,
                 weight_decay,
                 temp_dir,
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=False,
                 test_mode=False):

        assert epoch_idx < n_epoch,\
            'epoch_idx ("{}") must be lower than number of epochs ("{}")'.format(epoch_idx, n_epoch)

        self.n_epoch = n_epoch
        self.epoch_idx = epoch_idx
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.temp_dir = temp_dir
        self.checkpoint_freq = checkpoint_freq
        self.print_freq = print_freq
        self.use_progress_bar = use_progress_bar
        self.test_mode = test_mode

        self.metrics = ['cross_entropy', 'acc']
        self.monitor_metric = 'acc'
        self.monitor_direction = 'higher'

    def fit(self, model, train_loader, val_loader, test_loader, device, tensorboard_logger=None, logger_prefix=''):

        self.start_time = time.time()
        n_epoch_done = 0

        model.float()
        model.to(device)
        optimizer = self.get_optimizer(model)
        self.load_from_checkpoint(model, optimizer, device)

        while self.epoch_idx < self.n_epoch:
            # optimize one epoch
            self.optimize_epoch(model,
                                optimizer,
                                train_loader,
                                val_loader,
                                test_loader,
                                device)
            n_epoch_done += 1

            if self.print_freq > 0 and (self.epoch_idx + 1) % self.print_freq == 0:
                self.print_metrics(n_epoch_done)

            self.update_tensorboard(tensorboard_logger, logger_prefix)

            # save checkpoint
            if self.checkpoint_freq > 0 and (self.epoch_idx + 1) % self.checkpoint_freq == 0:
                checkpoint = {'epoch_idx': self.epoch_idx,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'metric_values': self.metric_values}

                checkpoint_file = os.path.join(self.temp_dir, 'checkpoint_{:09d}.pickle'.format(self.epoch_idx))
                fid = open(checkpoint_file, 'wb')
                pickle.dump(checkpoint, fid)
                fid.close()

            self.epoch_idx += 1

        # load the best model based on validation performance if exist, or train performance
        self.load_best(model)

        # return non-empty performance metrics
        performance = {}
        for metric in self.metric_values.keys():
            if len(self.metric_values[metric]) > 0:
                performance[metric] = self.metric_values[metric]

        return performance

    def load_best(self, model):
        # load the best model from checkpoints based on monitor_metric
        if len(self.metric_values['val_' + self.monitor_metric]) > 0:
            best_value = self.metric_values['val_' + self.monitor_metric][-1]
        else:
            best_value = self.metric_values['train_' + self.monitor_metric][-1]

        state_dict = model.state_dict()

        checkpoint_files = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir)
                            if f.startswith('checkpoint_')]

        for filename in checkpoint_files:
            fid = open(filename, 'rb')
            checkpoint = pickle.load(fid)
            fid.close()

            if len(checkpoint['metric_values']['val_' + self.monitor_metric]) > 0:
                metric_value = checkpoint['metric_values']['val_' + self.monitor_metric][-1]
            else:
                metric_value = checkpoint['metric_values']['train_' + self.monitor_metric][-1]

            if (self.monitor_direction == 'lower' and metric_value < best_value) or\
                    (self.monitor_direction == 'higher' and metric_value > best_value):
                best_value = metric_value
                state_dict = checkpoint['model_state_dict']

        model.load_state_dict(state_dict)

    def get_optimizer(self, model):
        assert self.optimizer in ['adam', 'sgd'], 'Given optimizer "{}" is not supported'.format(self.optimizer)

        # get current learning rate
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)

        # get separate batchnorm parameters and other parameters
        # if .get_parameters() is implemented in the model
        if hasattr(model, 'get_parameters') and callable(model.get_parameters):
            bn_params, other_params = model.get_parameters()

            if len(bn_params) > 0:
                params = [{'params': bn_params, 'weight_decay': 0},
                          {'params': other_params, 'weight_decay': self.weight_decay}]
            else:
                params = [{'params': other_params, 'weight_decay': self.weight_decay}]

            if self.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=lr)
            else:
                optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
        else:
            if self.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), weight_decay=self.weight_decay, lr=lr)
            else:
                optimizer = optim.SGD(model.parameters(),
                                      weight_decay=self.weight_decay,
                                      lr=lr,
                                      momentum=0.9,
                                      nesterov=True)

        return optimizer

    def eval(self, model, loader, device):
        if loader is None:
            return {}

        model.eval()

        L = torch.nn.CrossEntropyLoss()
        n_correct = 0
        n_sample = 0
        loss = 0

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (inputs, targets) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device).long().flatten()

                predictions = model(inputs)
                n_sample += inputs.size(0)
                loss += L(predictions, targets).item()
                n_correct += (predictions.argmax(dim=-1) == targets).sum().item()

        metrics = {'cross_entropy': loss / n_sample,
                   'acc': n_correct / n_sample}

        return metrics

    def update_lr(self,
                  optimizer):
        # update learning rate using lr_scheduler
        lr = self.lr_scheduler(self.n_epoch, self.epoch_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_loop(self, model, loader, optimizer, device):
        L = torch.nn.CrossEntropyLoss()

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        minibatch_idx = 0

        if self.use_progress_bar:
            loader = tqdm(loader, desc='#Epoch {}/{}: '.format(self.epoch_idx + 1, self.n_epoch), ncols=80, ascii=True)
        else:
            loader = loader

        for inputs, targets in loader:
            optimizer.zero_grad()
            self.update_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device).long().flatten()

            predictions = model(inputs)
            loss = L(predictions, targets)
            loss.backward()
            optimizer.step()

            minibatch_idx += 1

            if minibatch_idx > total_minibatch:
                break

    def update_metrics(self, train_metrics, val_metrics, test_metrics):
        for metric in train_metrics.keys():
            if 'train_' + metric in self.metric_values.keys():
                self.metric_values['train_' + metric].append(train_metrics[metric])
            else:
                self.metric_values['train_' + metric] = [train_metrics[metric]]

        for metric in val_metrics.keys():
            if 'val_' + metric in self.metric_values.keys():
                self.metric_values['val_' + metric].append(val_metrics[metric])
            else:
                self.metric_values['val_' + metric] = [val_metrics[metric]]

        for metric in test_metrics.keys():
            if 'test_' + metric in self.metric_values.keys():
                self.metric_values['test_' + metric].append(test_metrics[metric])
            else:
                self.metric_values['test_' + metric] = [test_metrics[metric]]

    def print_metrics(self, n_epoch_done):
        start_time = self.start_time
        current_time = time.time()
        n_epoch_remain = self.n_epoch - n_epoch_done

        # compute the time taken
        time_taken = current_time - start_time
        hour_taken = int(time_taken / 3600)
        minute_taken = int((time_taken - hour_taken * 3600) / 60)
        second_taken = int((time_taken - hour_taken * 3600 - minute_taken * 60))

        # compute estimated time remain
        time_left = (time_taken / n_epoch_done) * n_epoch_remain
        hour_left = int(time_left / 3600)
        minute_left = int((time_left - hour_left * 3600) / 60)
        second_left = int((time_left - hour_left * 3600 - minute_left * 60))

        msg = '#Epoch {}/{}, '.format(self.epoch_idx + 1, self.n_epoch) +\
            'total time taken: {:d}:{:02d}:{:02d}, '.format(hour_taken, minute_taken, second_taken) +\
            'time remain: {:d}:{:02d}:{:02d}'.format(hour_left, minute_left, second_left)

        print(msg)

        names = list(self.metric_values.keys())
        names.sort()
        for name in names:
            if len(self.metric_values[name]) > 0:
                value = self.metric_values[name][-1]
                if isinstance(value, (int, float)):
                    print('--- {}: {:.6f}'.format(name, value))

    def update_tensorboard(self, tensorboard_logger, logger_prefix):
        names = list(self.metric_values.keys())
        names.sort()
        if tensorboard_logger is not None:
            for name in names:
                if len(self.metric_values[name]) > 0:
                    value = self.metric_values[name][-1]
                    if isinstance(value, (int, float)):
                        tensorboard_logger.add_scalar(tag='{}/{}'.format(logger_prefix, name),
                                                      scalar_value=value,
                                                      global_step=self.epoch_idx + 1)
                        tensorboard_logger.flush()

    def optimize_epoch(self,
                       model,
                       optimizer,
                       train_loader,
                       val_loader,
                       test_loader,
                       device):
        model.train()

        # perform parameter updates
        self.update_loop(model, train_loader, optimizer, device)

        # evaluate
        train_metrics = self.eval(model, train_loader, device)
        val_metrics = self.eval(model, val_loader, device)
        test_metrics = self.eval(model, test_loader, device)

        # append current performance to performance list
        self.update_metrics(train_metrics, val_metrics, test_metrics)

    def load_from_checkpoint(self, model, optimizer, device):
        if self.epoch_idx == -1:
            # load from latest checkpoint
            files = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir)
                     if f.startswith('checkpoint_')]
            files.sort()

            if len(files) > 0:
                fid = open(files[-1], 'rb')
                checkpoint = pickle.load(fid)
                fid.close()
            else:
                checkpoint = None
        elif self.epoch_idx == 0:
            # train from scratch
            checkpoint = None
        else:
            # load specific checkpoint
            filename = os.path.join(self.temp_dir, 'checkpoint_{:9d}.pickle'.format(self.epoch_idx))
            assert os.path.exists(filename),\
                'checkpoint "{}" does not exist'.format(filename)

            fid = open(filename, 'rb')
            checkpoint = pickle.load(fid)
            fid.close()

        if checkpoint is None:
            self.epoch_idx = 0
            self.metric_values = {}
            prefixes = ['train', 'val', 'test']
            for prefix in prefixes:
                for m in self.metrics:
                    self.metric_values['{}_{}'.format(prefix, m)] = []
        else:
            # set the epoch index and previous metric values
            self.epoch_idx = checkpoint['epoch_idx'] + 1
            self.metric_values = checkpoint['metric_values']

            # load model state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            # load optimizer state dict
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)


class AutoRegressionTrainer(ClassifierTrainer):
    def __init__(self,
                 n_epoch,
                 epoch_idx,
                 lr_scheduler,
                 optimizer,
                 weight_decay,
                 temp_dir,
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=False,
                 test_mode=False):

        super(AutoRegressionTrainer, self).__init__(n_epoch, epoch_idx, lr_scheduler, optimizer, weight_decay,
                                                    temp_dir, checkpoint_freq, print_freq, use_progress_bar, test_mode)

        self.metrics = ['mean_squared_error']
        self.monitor_metric = 'mean_squared_error'
        self.monitor_direction = 'lower'

    def update_loop(self, model, loader, optimizer, device):
        L = torch.nn.MSELoss()

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        minibatch_idx = 0

        if self.use_progress_bar:
            loader = tqdm(loader, desc='#Epoch {}/{}: '.format(self.epoch_idx + 1, self.n_epoch), ncols=80, ascii=True)
        else:
            loader = loader

        for inputs, _ in loader:

            optimizer.zero_grad()
            self.update_lr(optimizer)

            inputs = inputs.to(device)

            predictions = model(inputs)
            loss = L(predictions, inputs)
            loss.backward()
            optimizer.step()

            minibatch_idx += 1
            if minibatch_idx > total_minibatch:
                break

    def eval(self, model, loader, device):
        if loader is None:
            return {}

        model.eval()

        L = torch.nn.MSELoss()
        n_sample = 0
        loss = 0

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (inputs, _) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                inputs = inputs.to(device)

                predictions = model(inputs)
                n_sample += inputs.size(0)
                loss += L(predictions, inputs).item()

        metrics = {'mean_squared_error': loss / n_sample}

        return metrics
