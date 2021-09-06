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

# general imports
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# OpenDR imports
from opendr.perception.compressive_learning.multilinear_compressive_learning.algorithm.trainers import\
    ClassifierTrainer as BaseTrainer


class ClassifierTrainer(BaseTrainer):
    def __init__(self,
                 n_epoch,
                 epoch_idx,
                 lr_scheduler,
                 optimizer,
                 weight_decay,
                 temp_dir,
                 class_weight=None,
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=False,
                 test_mode=False):

        super(ClassifierTrainer, self).__init__(n_epoch=n_epoch,
                                                epoch_idx=epoch_idx,
                                                lr_scheduler=lr_scheduler,
                                                optimizer=optimizer,
                                                weight_decay=weight_decay,
                                                temp_dir=temp_dir,
                                                checkpoint_freq=checkpoint_freq,
                                                print_freq=print_freq,
                                                use_progress_bar=use_progress_bar,
                                                test_mode=test_mode)

        self.metrics.extend(['precision', 'recall', 'f1'])
        self.monitor_metric = 'f1'
        self.monitor_direction = 'higher'
        self.class_weight = class_weight

    def eval(self, model, loader, device):
        if loader is None:
            return {}

        model.eval()

        L = torch.nn.CrossEntropyLoss()
        n_sample = 0
        loss = 0
        y_true = []
        y_pred = []

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
                y_true.extend(targets.flatten().tolist())
                y_pred.extend(predictions.argmax(dim=-1).flatten().tolist())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        metrics = {'cross_entropy': loss / n_sample,
                   'acc': acc,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1}

        return metrics

    def update_loop(self, model, loader, optimizer, device):
        if self.class_weight is None:
            L = torch.nn.CrossEntropyLoss()
        else:
            L = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight, device=device).float())

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
