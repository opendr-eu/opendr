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
from torch import nn
import torch.optim as optim
from torch.nn import functional
import numpy as np

torch.manual_seed(2605)
torch.cuda.manual_seed_all(2605)


class BhattacharyyaCoeff(nn.Module):
    def __init__(self, input_dim=256):
        super(BhattacharyyaCoeff, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, input_dim, 1, 1, 1))

    def forward(self, z, x):
        N = z.size(0)
        C = z.size(1)
        k = z.size(2)
        m = x.size(2)
        N_b = (m - k + 1) ** 2
        x_unf = functional.unfold(x, k).view(N, C, k, k, N_b)

        coeff = torch.sum(
            torch.sqrt(x_unf * z.unsqueeze_(4).repeat(1, 1, 1, 1, N_b)) *
            self.weights,
            dim=1,
            keepdim=True,
        )
        mean_coeff = (
            coeff.mean(dim=2).mean(dim=2).view(N, 1, (m - k + 1), (m - k + 1))
        )
        return mean_coeff


class Adjust2d(nn.Module):
    def __init__(self):
        super(Adjust2d, self).__init__()
        self.type = "bn"
        self.bn = nn.BatchNorm2d(1)
        self._initialize_weights()

    def forward(self, input):
        out = self.bn(input)
        return out

    def _initialize_weights(self):
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()


class SiameseBhatNet(nn.Module):
    def __init__(self, branch):
        super(SiameseBhatNet, self).__init__()
        self.branch = branch
        self.join = BhattacharyyaCoeff(self.branch.output_dim)
        self.norm = Adjust2d()

    def forward(self, z, x):
        x = self.branch(x)
        z = self.branch(z)
        out = self.join(z, x)
        out = self.norm(out)
        return out, x, z

    def process_features(self, z, x):
        out = self.join(z, x)
        out = self.norm(out)

        return out


class SiameseConvNet(nn.Module):
    def __init__(self, branch):
        super(SiameseConvNet, self).__init__()
        self.branch = branch
        self.join = functional.conv2d
        self.norm = Adjust2d()

    def forward(self, z, x):
        x = self.branch(x)
        z = self.branch(z)
        out = self.process_features(z, x)

        return out, x, z

    def process_features(self, z, x):
        out = self.join(z, x)
        out = self.norm(out)

        return out


class CHNet(nn.Module):
    def __init__(self, voxelnet, exp_config):
        super(CHNet, self).__init__()
        self.cfg = exp_config.model_config
        self.train_cfg = exp_config.train_config
        self.device = self.cfg.device

        branch = voxelnet
        self.model = SiameseConvNet(branch)

        self.model = self.model.cuda()
        self.initialized = False
        self.output_sz, self.total_stride = self._deduce_network_params(
            self.cfg.template_sz, self.cfg.search_sz
        )

    def forward(self, input_z, input_x):
        return self.model(input_z, input_x)

    def setup_finetuning_params(self):
        self.params = []
        for name, param in self.model.named_parameters():
            weight_decay = self.train_cfg.weight_decay
            if "bof" in name or "join" in name or "norm" in name:
                lr = self.train_cfg.finetune_lr
                self.params.append(
                    {
                        "params": param,
                        "initial_lr": lr,
                        "weight_decay": weight_decay,
                        "name": name,
                    }
                )
        self.setup_optimizer()

    def setup_params(self):
        self.params = []
        for name, param in self.model.named_parameters():
            lr = self.train_cfg.initial_lr
            weight_decay = self.train_cfg.weight_decay
            if "bof" in name:
                lr *= 2.0
            self.params.append(
                {
                    "params": param,
                    "initial_lr": lr,
                    "weight_decay": weight_decay,
                    "name": name,
                }
            )

    def calc_score(self, z, x):
        with torch.set_grad_enabled(False):
            self.model.norm.eval()
            scores = self.model.join(z.repeat(3, 1, 1, 1), x)
            scores = self.model.norm(scores)
        return scores

    def setup_optimizer(self):
        if self.train_cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.params,
                lr=self.train_cfg.initial_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        elif self.train_cfg.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.params,
                lr=self.train_cfg.initial_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        gamma = self.train_cfg.lr_decay_gamma
        self.scheduler = torch.nn.StepLR(
            self.optimizer, self.train_cfg.step_size, gamma=gamma
        )
        if self.train_cfg.loss == "bce":
            self.criterion = torch.nn.BCEWeightedLoss()
        elif self.train_cfg.loss == "focal":
            self.criterion = torch.nn.FocalLoss(alpha=0, gamma=1)  # .to(self.device)
        elif self.train_cfg.loss == "softmargin":
            self.criterion = torch.nn.SoftMarginLoss()
        print(self.criterion)

    def step(self, batch, backward=True, update_lr=True, step=None):
        if backward:
            if update_lr:
                self.scheduler.step()
                # print 'updating lr with gamma=...', self.scheduler.gamma, self.scheduler.get_lr()[0]
            self.model.train()
        else:
            self.model.eval()

        z, x = batch[0].cuda(), batch[1].cuda()
        t, w = batch[2].cuda(), batch[3].cuda()

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(backward):
            pred, feat_x, feat_z = self.model(z, x)
            # if self.train_cfg.max_pooling:
            #     pred = self.model.pool(pred)
            if self.train_cfg.labels == "gaussian":
                loss = self.criterion(pred, self.labels)
            else:
                loss = self.criterion(pred, t, w)
            if self.train_cfg.use_regularizer:
                loss += 0.001 * self.regularizer(feat_x, feat_z)
            if backward:
                loss.backward()
                self.optimizer.step()

        return loss.item(), pred

    def _create_labels(self, n=None):
        if self.train_cfg.labels == "binary":
            labels = self._create_logisticloss_labels()
            weights = np.zeros_like(labels)

            pos_num = np.sum(labels == 1)
            neg_num = np.sum(labels == 0)
            weights[labels == 1] = 0.5 / pos_num
            weights[labels == 0] = 0.5 / neg_num
            weights *= pos_num + neg_num

            labels = labels[np.newaxis, :]
            weights = weights[np.newaxis, :]

            labels = torch.from_numpy(labels).float()
            weights = torch.from_numpy(weights).float()
        else:
            labels = self._create_gaussian_labels().float()
            weights = torch.ones(labels.size()).float()

        if n is None:
            labels = labels.unsqueeze_(0).repeat(
                self.train_cfg.batch_size, 1, 1, 1
            )
            weights = weights.unsqueeze_(0).repeat(
                self.train_cfg.batch_size, 1, 1, 1
            )
        else:
            labels = labels.unsqueeze_(0).repeat(n, 1, 1, 1)
            weights = weights.unsqueeze_(0).repeat(n, 1, 1, 1)

        return labels, weights

    def _create_logisticloss_labels(self):
        label_sz = self.output_sz
        r_pos = self.train_cfg.r_pos / self.total_stride
        r_neg = self.train_cfg.r_neg / self.total_stride
        labels = np.zeros((label_sz, label_sz))

        for r in range(label_sz):
            for c in range(label_sz):
                dist = np.sqrt(
                    (r - label_sz // 2) ** 2 + (c - label_sz // 2) ** 2
                )
                if dist <= r_pos:
                    labels[r, c] = 1
                elif dist <= r_neg:
                    labels[r, c] = self.train_cfg.ignore_label
                else:
                    labels[r, c] = 0
        return labels

    def _deduce_network_params(self, exemplar_sz, search_sz):
        z = torch.ones(1, 3, int(exemplar_sz), int(exemplar_sz)).cuda()
        x = torch.ones(1, 3, int(search_sz), int(search_sz)).cuda()
        with torch.no_grad():
            self.model.eval()
            # y, _, _ = self.model(z, x, join='xcorr')
            x, y, z = self.model.get_xyz(z, x)
        score_sz = y.size(-1)
        print("Output size : ", score_sz)
        print("z size : ", z.size())
        print("x size : ", x.size())

        total_stride = 1
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
                stride = (
                    m.stride[0] if isinstance(m.stride, tuple) else m.stride
                )
                total_stride *= stride

        print("Total stride : ", total_stride)
        # print 'Receptive field z : ', compute_RF_numerical(self.model.branch, z, device=self.device)
        # print 'Receptive field x : ', compute_RF_numerical(self.model.branch, x, device=self.device)

        return score_sz, total_stride
