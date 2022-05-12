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

"""
Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1–1, New York, USA.
"""

# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import PIL
import numpy as np
import torch
from torch.autograd import Variable
import os
from os import path, makedirs
import onnxruntime
from typing import Union, List
import shutil
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.perception.facial_expression_recognition.ensemble_based_cnn.algorithm.model.esr_9 import ESR
from opendr.perception.facial_expression_recognition.ensemble_based_cnn.algorithm.utils import datasets, plotting


class EnsembleCNNLearner(Learner):
    def __init__(self, lr=1e-1, batch_size=32, lr_schedule='',
                 temp_path='./tmp/', device='cuda', device_ind=[0],
                 validation_interval=1, max_training_epoch=2, momentum=0.9,
                 ensemble_size=9, base_path_experiment='./experiments/', name_experiment='AffectNet_Continuous',
                 dimensional_finetune=True, categorical_train=False, base_path_to_dataset='./data/AffectNet',
                 max_tuning_epoch=1
                 ):
        super(EnsembleCNNLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                 temp_path=temp_path, device=device)
        # dataset_name = ''
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.device_ind = device_ind
        self.output_device = self.device_ind[0] if type(self.device_ind) is list else self.device_ind
        self.parent_dir = temp_path
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.temp_path = temp_path
        self.lr_schedule = lr_schedule
        self.base_path_experiment = base_path_experiment
        self.name_experiment = name_experiment
        self.base_path_to_dataset = base_path_to_dataset
        self.validation_interval = validation_interval
        self.max_training_epoch = max_training_epoch
        self.ensemble_size = ensemble_size
        self.dimensional_finetune = dimensional_finetune
        self.categorical_train = categorical_train
        self.ort_session = None
        self.max_tuning_epoch = max_tuning_epoch

    def init_model(self, num_branches):
        self.model = ESR(device=self.device, ensemble_size=num_branches)
        self.model.to_device(self.device)

    def save(self, state_dicts, base_path_to_save_model, current_branch_save):
        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}}
        if not path.isdir(path.join(base_path_to_save_model, str(current_branch_save))):
            makedirs(path.join(base_path_to_save_model, str(current_branch_save)))
        if self.ort_session is None:
            model_metadata["model_paths"] = [path.join(base_path_to_save_model, str(current_branch_save))]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pt"
            torch.save(state_dicts[0],
                       path.join(base_path_to_save_model, str(current_branch_save),
                                 "Net-Base-Shared_Representations.pt"))
            for i in range(1, len(state_dicts)):
                torch.save(state_dicts[i],
                           path.join(base_path_to_save_model, str(current_branch_save), "Net-Branch_{}.pt".format(i)))
            print("Pytorch model has been "
                  "saved at: {}".format(path.join(base_path_to_save_model, str(current_branch_save))))
        else:
            model_metadata["model_paths"] = [[path.join(base_path_to_save_model, str(current_branch_save))]]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"
            shutil.copy2(path.join(self.temp_path, self.name_experiment, "onnx_model.onnx"),
                         model_metadata["model_paths"][0])
            print("ONNX model has been "
                  "saved at: {}".format(path.join(base_path_to_save_model, str(current_branch_save))))
        json_model_name = self.name_experiment + '.json'
        json_model_path = path.join(base_path_to_save_model, str(current_branch_save), json_model_name)
        with open(json_model_path, 'w') as outfile:
            json.dump(model_metadata, outfile)

    def load(self, ensemble_size, path_to_saved_network="./trained_models/esr_9",
             file_name_base_network="Net-Base-Shared_Representations.pt",
             file_name_conv_branch="Net-Branch_{}.pt", fix_backbone=True):
        with open(path.join(path_to_saved_network, self.name_experiment + ".json")) as metadata_file:
            metadata = json.load(metadata_file)
        if metadata["optimized"]:
            self.__load_from_onnx(path.join(path_to_saved_network, self.name_experiment + '.onnx'))
        else:
            # load base
            self.model.base.load_state_dict(torch.load(
                path.join(path_to_saved_network, file_name_base_network), map_location=self.device))
            # Load branches
            for i in range(ensemble_size):
                self.model.convolutional_branches[i].load_state_dict(
                    torch.load(path.join(path_to_saved_network, file_name_conv_branch.format(i + 1)),
                               map_location=self.device))
            if self.dimensional_finetune and fix_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                for i in range(ensemble_size):
                    for p in self.model.convolutional_branches[i].fc_dimensional.parameters():
                        p.requires_grad = True

                '''# Base no trainable
                for module in self.model.base:
                    for p in module.parameters():
                        p.requires_grad = False
                # Branch no trainable, but last layer
                for i in range(ensemble_size):
                    for module in self.model.convolutional_branches[i]:
                        for p in module.parameters():
                            p.requires_grad = False
                    for p in self.model.convolutional_branches[i].fc_dimensional.parameters():
                        p.requires_grad = True'''

    def fit(self):
        # Make dir
        if not path.isdir(path.join(self.base_path_experiment, self.name_experiment)):
            makedirs(path.join(self.base_path_experiment, self.name_experiment))
        # Define data transforms
        data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.RandomAffine(degrees=30,
                                                   translate=(.1, .1),
                                                   scale=(1.0, 1.25),
                                                   resample=PIL.Image.BILINEAR)]
        print("Starting: {}".format(str(self.name_experiment)))
        print("Running on {}".format(self.device))

        # Train a new model on AffectNet_Categorical from scratch
        if self.categorical_train:
            self.model = None
            self.init_model(num_branches=1)  # The model is built by adding and training branches one by one
            self.model.to_device(self.device)
            self.criterion_cat = nn.CrossEntropyLoss()
            self.optimizer_ = optim.SGD([{'params': self.model.base.parameters(), 'lr': self.lr,
                                         'momentum': self.momentum},
                                        {'params': self.model.convolutional_branches[-1].parameters(), 'lr': self.lr,
                                         'momentum': self.momentum}])
            # Data loader
            train_data = datasets.AffectNetCategorical(idx_set=0,
                                                       max_loaded_images_per_label=5000,
                                                       transforms=transforms.Compose(data_transforms),
                                                       is_norm_by_mean_std=False,
                                                       base_path_to_affectnet=self.base_path_to_dataset)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
            val_data = datasets.AffectNetCategorical(idx_set=2,
                                                     max_loaded_images_per_label=100000,
                                                     transforms=None,
                                                     is_norm_by_mean_std=False,
                                                     base_path_to_affectnet=self.base_path_to_dataset)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=8)

            for branch_on_training in range(self.ensemble_size):
                branch_on_training
                # Best network
                best_ensemble = self.model.to_state_dict()
                best_ensemble_acc = 0.0
                # Initialize scheduler
                scheduler = optim.lr_scheduler.StepLR(self.optimizer_, step_size=10, gamma=0.5, last_epoch=-1)
                # History
                history_loss = []
                history_acc = [[] for _ in range(self.model.get_ensemble_size())]
                history_val_loss = [[] for _ in range(self.model.get_ensemble_size())]
                history_val_acc = [[] for _ in range(self.model.get_ensemble_size() + 1)]

                # Training branch
                for epoch in range(self.max_training_epoch):
                    running_loss = 0.0
                    running_corrects = [0.0 for _ in range(self.model.get_ensemble_size())]
                    running_updates = 0
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer_.zero_grad()
                        # Forward
                        out_emotions, out_va = self.model(inputs)
                        confs_preds = [torch.max(o, 1) for o in out_emotions]
                        # Compute loss
                        loss = 0.0
                        for i_4 in range(self.model.get_ensemble_size()):
                                         preds = confs_preds[i_4][1]
                                         running_corrects[i_4] += torch.sum(preds == labels).cpu().numpy()
                                         loss += self.criterion_cat(out_emotions[i_4], labels)
                        # Backward
                        loss.backward()
                        # Optimize
                        self.optimizer_.step()
                        scheduler.step()
                        # Save loss
                        running_loss += loss.item()
                        running_updates += 1
                    # Statistics
                    print('[Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.
                          format(self.model.get_ensemble_size(), epoch+1, self.max_training_epoch,
                                 running_loss / running_updates, np.array(running_corrects) / len(train_data)))

                    # Validation
                    if ((epoch % self.validation_interval) == 0) or ((epoch + 1) == self.max_training_epoch):
                        self.model.eval()
                        eval_results = self.eval(eval_type='categorical', current_branch_on_training=branch_on_training)
                        val_loss = eval_results["running_emotion_loss"]
                        val_corrects = eval_results["running_emotion_corrects"]

                        print('Validation - [Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}'.format(
                               self.model.get_ensemble_size(), epoch + 1, self.max_training_epoch, val_loss[-1],
                               np.array(val_corrects) / len(val_data)))

                        # Add to history training and validation statistics
                        history_loss.append(running_loss / running_updates)
                        for b in range(self.model.get_ensemble_size()):
                            history_acc[b].append(running_corrects[b] / len(train_data))
                            history_val_loss[b].append(val_loss[b])
                            history_val_acc[b].append(float(val_corrects[b]) / len(val_data))

                        # Add ensemble accuracy to history
                        history_val_acc[-1].append(float(val_corrects[-1]) / len(val_data))
                        # Save best ensemble
                        ensemble_acc = (float(val_corrects[-1]) / len(val_data))
                        if ensemble_acc >= best_ensemble_acc:
                            best_ensemble_acc = ensemble_acc
                            best_ensemble = self.model.to_state_dict()
                            # Save network
                            self.save(best_ensemble,
                                      path.join(self.base_path_experiment, self.name_experiment, 'Saved_Networks'),
                                      self.model.get_ensemble_size())

                        # Save graphs
                        self.__plot_categorical(history_loss, history_acc, history_val_loss, history_val_acc,
                                                self.model.get_ensemble_size(),
                                                path.join(self.base_path_experiment, self.name_experiment))
                        # Set network to training mode
                        self.model.train()

                # Change branch on training
                if self.model.get_ensemble_size() < self.ensemble_size:
                    self.max_training_epoch = self.max_tuning_epoch
                    # Reload best configuration
                    self.model.reload(best_ensemble)
                    # Add a new branch
                    self.model.add_branch()
                    self.model.to_device(self.device)
                    self.optimizer_ = optim.SGD([{'params': self.model.base.parameters(), 'lr': self.lr/10,
                                            'momentum': self.momentum},
                                           {'params': self.model.convolutional_branches[-1].parameters(), 'lr': self.lr,
                                            'momentum': self.momentum}])
                    for b in range(self.model.get_ensemble_size() - 1):
                        self.optimizer_.add_param_group({'params': self.model.convolutional_branches[b].parameters(),
                                                   'lr': self.lr/10, 'momentum': self.momentum})
                # Finish training after training all branches
                else:
                    break

        # Finetune the trained model on AffectNet_dimensional dataset for VA-estimation
        if self.dimensional_finetune:
            self.init_model(num_branches=self.ensemble_size)
            # Load network trained on AffectNet_Categorical and fix its backbone
            self.load(self.ensemble_size, path_to_saved_network=path.join(
                self.base_path_experiment, self.name_experiment, 'Saved_Networks', str(self.model.get_ensemble_size())),
                fix_backbone=True)
                        # Set loss and optimizer
            self.model.to_device(self.device)
            self.criterion_dim = nn.MSELoss(reduction='mean')
            self.optimizer_ = optim.SGD([{'params': self.model.base.parameters(), 'lr': self.lr,
                                         'momentum': self.momentum},
                                        {'params': self.model.convolutional_branches[0].parameters(),
                                         'lr': self.lr, 'momentum': self.momentum}])
            for b in range(1, self.model.get_ensemble_size()):
                self.optimizer_.add_param_group({'params': self.model.convolutional_branches[b].parameters(),
                                                'lr': self.lr / 10, 'momentum': self.momentum})
            # Data loaders
            train_data = datasets.AffectNetDimensional(idx_set=0,
                                                       max_loaded_images_per_label=5000,
                                                       transforms=transforms.Compose(data_transforms),
                                                       is_norm_by_mean_std=False,
                                                       base_path_to_affectnet=self.base_path_to_dataset)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=8)

            # Finetune the pretrained model on continuous affect values
            self.__finetune(train_loader=train_loader)

    def __finetune(self, train_loader):
        current_branch_on_training = 0
        for branch_on_training in range(self.ensemble_size):
            # Best network
            best_ensemble = self.model.to_state_dict()
            best_ensemble_rmse = 10000000.0
            # History
            history_loss = []
            history_val_loss_valence = [[] for _ in range(self.model.get_ensemble_size() + 1)]
            history_val_loss_arousal = [[] for _ in range(self.model.get_ensemble_size() + 1)]

            # Training branch
            for epoch in range(self.max_training_epoch):
                running_loss = 0.0
                running_updates = 0
                batch = 0
                for inputs, labels in train_loader:
                    batch += 1
                    # Get the inputs
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels_valence = labels[:, 0].view(len(labels[:, 0]), 1)
                    labels_arousal = labels[:, 1].view(len(labels[:, 1]), 1)
                    self.optimizer_.zero_grad()
                    # Forward
                    out_emotions, out_va = self.model(inputs)
                    # Compute loss of affect_values
                    loss = 0.0
                    for i_4 in range(current_branch_on_training + 1):
                        out_valence = out_va[i_4][:, 0].view(len(out_va[i_4][:, 0]), 1)
                        out_arousal = out_va[i_4][:, 1].view(len(out_va[i_4][:, 1]), 1)
                        loss += torch.sqrt(self.criterion_dim(out_valence, labels_valence))
                        loss += torch.sqrt(self.criterion_dim(out_arousal, labels_arousal))
                    # Backward
                    loss.backward()
                    # Optimize
                    self.optimizer_.step()
                    # Save loss
                    running_loss += loss.item()
                    running_updates += 1
                # Statistics
                print('[Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f}'.
                      format(current_branch_on_training + 1, epoch + 1, self.max_training_epoch,
                             running_loss / running_updates))
                # Validation
                if (epoch % self.validation_interval) == 0:
                    self.model.eval()
                    eval_results = self.eval(eval_type='dimensional',
                                             current_branch_on_training=current_branch_on_training)
                    val_loss = eval_results["valence_arousal_losses"]
                    # Add to history training and validation statistics
                    history_loss.append(running_loss / running_updates)
                    for b in range(self.model.get_ensemble_size()):
                        history_val_loss_valence[b].append(val_loss[0][b])
                        history_val_loss_arousal[b].append(val_loss[1][b])

                    # Add ensemble rmse to history
                    history_val_loss_valence[-1].append(val_loss[0][-1])
                    history_val_loss_arousal[-1].append(val_loss[1][-1])
                    print('Validation - [Branch {:d}, Epochs {:d}--{:d}] Loss (V) - (A): ({}) - ({})'.format(
                          current_branch_on_training + 1, epoch + 1, self.max_training_epoch,
                          [hvlv[-1] for hvlv in history_val_loss_valence],
                          [hvla[-1] for hvla in history_val_loss_arousal]))

                    # Save best ensemble
                    ensemble_rmse = float(history_val_loss_valence[-1][-1]) + float(history_val_loss_arousal[-1][-1])
                    if ensemble_rmse <= best_ensemble_rmse:
                        best_ensemble_rmse = ensemble_rmse
                        best_ensemble = self.model.to_state_dict()
                        # Save network
                        self.save(best_ensemble, path.join(self.base_path_experiment,
                                                           self.name_experiment, 'Saved_Networks'),
                                                           current_branch_on_training + 1)

                    # Save graphs
                    self.__plot_dimensional(history_loss, history_val_loss_valence, history_val_loss_arousal,
                                            current_branch_on_training + 1,
                                            path.join(self.base_path_experiment, self.name_experiment))
                    self.model.train()

            # Change branch on training
            if current_branch_on_training < self.model.get_ensemble_size:
                current_branch_on_training += 1
                self.max_training_epoch = 2
                # Reload best configuration
                self.model.reload(best_ensemble)
                self.model.to_device(self.device)
                for b in range(self.model.get_ensemble_size()):
                    self.optimizer_.add_param_group({'params': self.model.convolutional_branches[b].parameters(),
                                                    'lr': self.lr / 10,
                                                    'momentum': self.momentum})
                self.optimizer_ = optim.SGD([{'params': self.model.base.parameters(), 'lr': self.lr / 10,
                                             'momentum': self.momentum},
                                            {'params': self.model.convolutional_branches
                                            [current_branch_on_training].parameters(),
                                             'lr': self.lr,
                                             'momentum': self.momentum}])
            # Finish training after fine-tuning all branches
            else:
                break

    def eval(self, eval_type='categorical', current_branch_on_training=0):
        cpu_device = torch.device('cpu')
        val_va_predictions = [[] for _ in range(self.model.get_ensemble_size() + 1)]
        val_targets_valence = []
        val_targets_arousal = []
        valence_arousal_losses = [[], []]

        running_emotion_loss = [0.0 for _ in range(self.model.get_ensemble_size())]
        running_emotion_corrects = [0 for _ in range(self.model.get_ensemble_size() + 1)]
        running_emotion_steps = [0 for _ in range(self.model.get_ensemble_size())]

        if eval_type == 'categorical':
            # load data
            val_data = datasets.AffectNetCategorical(idx_set=2,
                                                     max_loaded_images_per_label=100000,
                                                     transforms=None,
                                                     is_norm_by_mean_std=False,
                                                     base_path_to_affectnet=self.base_path_to_dataset)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=8)
            # evaluate
            for inputs_eval, labels_eval in val_loader:
                inputs_eval, labels_eval = inputs_eval.to(self.device), labels_eval.to(self.device)
                out_emotion_eval, out_va_eval = self.model(inputs_eval)
                outputs_eval = out_emotion_eval[:current_branch_on_training + 1]
                # Ensemble prediction
                overall_preds = torch.zeros(outputs_eval[0].size()).to(self.device)
                for o_eval, outputs_per_branch_eval in enumerate(outputs_eval, 0):
                    _, preds_eval = torch.max(outputs_per_branch_eval, 1)
                    running_emotion_corrects[o_eval] += torch.sum(preds_eval == labels_eval).cpu().numpy()
                    loss_eval = self.criterion_cat(outputs_per_branch_eval, labels_eval)
                    running_emotion_loss[o_eval] += loss_eval.item()
                    running_emotion_steps[o_eval] += 1
                    for v_i, v_p in enumerate(preds_eval, 0):
                        overall_preds[v_i, v_p] += 1
                # Compute accuracy of ensemble predictions
                _, preds_eval = torch.max(overall_preds, 1)
                running_emotion_corrects[-1] += torch.sum(preds_eval == labels_eval).cpu().numpy()

            for b_eval in range(self.model.get_ensemble_size()):
                div = running_emotion_steps[b_eval] if running_emotion_steps[b_eval] != 0 else 1
                running_emotion_loss[b_eval] /= div

        elif eval_type == 'dimensional':
            # load data
            val_data = datasets.AffectNetDimensional(idx_set=2,
                                                     max_loaded_images_per_label=100000,
                                                     transforms=None,
                                                     is_norm_by_mean_std=False,
                                                     base_path_to_affectnet=self.base_path_to_dataset)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=8)
            # evaluate model
            for inputs_eval, labels_eval in val_loader:
                inputs_eval, labels_eval = inputs_eval.to(self.device), labels_eval
                labels_eval_valence = labels_eval[:, 0].view(len(labels_eval[:, 0]), 1)
                labels_eval_arousal = labels_eval[:, 1].view(len(labels_eval[:, 1]), 1)
                out_emotion_eval, out_va_eval = self.model(inputs_eval)
                outputs_eval = out_va_eval[:current_branch_on_training + 1]

                # Ensemble prediction
                val_predictions_ensemble = torch.zeros(outputs_eval[0].size()).to(cpu_device)
                for evaluate_branch in range(current_branch_on_training + 1):
                    out_va_eval_cpu = out_va_eval[evaluate_branch].detach().to(cpu_device)
                    val_va_predictions[evaluate_branch].extend(out_va_eval_cpu)
                    val_predictions_ensemble += out_va_eval_cpu
                val_va_predictions[-1].extend(val_predictions_ensemble / (current_branch_on_training + 1))
                val_targets_valence.extend(labels_eval_valence)
                val_targets_arousal.extend(labels_eval_arousal)

            val_targets_valence = torch.stack(val_targets_valence)
            val_targets_arousal = torch.stack(val_targets_arousal)

            for evaluate_branch in range(self.model.get_ensemble_size() + 1):
                if evaluate_branch < (current_branch_on_training + 1) or \
                        evaluate_branch == self.model.get_ensemble_size():
                    list_tensor = torch.stack(val_va_predictions[evaluate_branch])
                    out_valence_eval = list_tensor[:, 0].view(len(list_tensor[:, 0]), 1)
                    out_arousal_eval = list_tensor[:, 1].view(len(list_tensor[:, 1]), 1)
                    valence_arousal_losses[0].append(torch.sqrt(self.criterion_dim(out_valence_eval,
                                                                                   val_targets_valence)))
                    valence_arousal_losses[1].append(torch.sqrt(self.criterion_dim(out_arousal_eval,
                                                                                   val_targets_arousal)))
                else:
                    valence_arousal_losses[0].append(torch.tensor(0))
                    valence_arousal_losses[1].append(torch.tensor(0))
        results = {
            "valence_arousal_losses": valence_arousal_losses,
            "running_emotion_loss": running_emotion_loss,
            "running_emotion_corrects": running_emotion_corrects
        }
        return results

    def __plot_dimensional(self, his_loss, his_val_loss_valence, his_val_loss_arousal, branch_idx, base_path_his):
        losses_plot = [[range(len(his_loss)), his_loss]]
        legends_plot_loss = ['Training']
        # Loss
        for b_plot in range(len(his_val_loss_valence)):
            losses_plot.append([range(len(his_val_loss_valence[b_plot])), his_val_loss_valence[b_plot]])
            legends_plot_loss.append('Validation ({}) (Val)'.format(b_plot + 1))
            losses_plot.append([range(len(his_val_loss_arousal[b_plot])), his_val_loss_arousal[b_plot]])
            legends_plot_loss.append('Validation ({}) (Aro)'.format(b_plot + 1))

        # Loss
        plotting.plot(losses_plot,
                      title='Training and Validation Losses vs. Epochs for Branch {}'.format(branch_idx),
                      legends=legends_plot_loss,
                      file_path=base_path_his,
                      file_name='Loss_Branch_{}'.format(branch_idx),
                      axis_x='Training Epoch',
                      axis_y='Loss',
                      limits_axis_y=(0.2, 0.6, 0.025))

        np.save(path.join(base_path_his, 'Loss_Branch_{}'.format(branch_idx)), np.array(his_loss))
        np.save(path.join(base_path_his, 'Loss_Val_Branch_{}_Valence'.format(branch_idx)), np.array(his_val_loss_valence))
        np.save(path.join(base_path_his, 'Loss_Val_Branch_{}_Arousal'.format(branch_idx)), np.array(his_val_loss_arousal))

    def __plot_categorical(self, his_loss, his_acc, his_val_loss, his_val_acc, branch_idx, base_path_his):
        accuracies_plot = []
        legends_plot_acc = []
        losses_plot = [[range(len(his_loss)), his_loss]]
        legends_plot_loss = ["Training"]

        # Acc
        for b_plot in range(len(his_acc)):
            accuracies_plot.append([range(len(his_acc[b_plot])), his_acc[b_plot]])
            legends_plot_acc.append("Training ({})".format(b_plot + 1))
            accuracies_plot.append([range(len(his_val_acc[b_plot])), his_val_acc[b_plot]])
            legends_plot_acc.append("Validation ({})".format(b_plot + 1))

        # Ensemble acc
        accuracies_plot.append([range(len(his_val_acc[-1])), his_val_acc[-1]])
        legends_plot_acc.append("Validation (E)")

        # Accuracy
        plotting.plot(accuracies_plot,
                      title="Training and Validation Accuracies vs. Epochs for Branch {}".format(branch_idx),
                      legends=legends_plot_acc,
                      file_path=base_path_his,
                      file_name="Acc_Branch_{}".format(branch_idx),
                      axis_x="Training Epoch",
                      axis_y="Accuracy",
                      limits_axis_y=(0.0, 1.0, 0.025))

        # Loss
        for b_plot in range(len(his_val_loss)):
            losses_plot.append([range(len(his_val_loss[b_plot])), his_val_loss[b_plot]])
            legends_plot_loss.append("Validation ({})".format(b_plot + 1))
        plotting.plot(losses_plot,
                      title="Training and Validation Losses vs. Epochs for Branch {}".format(branch_idx),
                      legends=legends_plot_loss,
                      file_path=base_path_his,
                      file_name="Loss_Branch_{}".format(branch_idx),
                      axis_x="Training Epoch",
                      axis_y="Loss")

        # Save plots
        np.save(path.join(base_path_his, "Loss_Branch_{}".format(branch_idx)), np.array(his_loss))
        np.save(path.join(base_path_his, "Acc_Branch_{}".format(branch_idx)), np.array(his_acc))
        np.save(path.join(base_path_his, "Loss_Val_Branch_{}".format(branch_idx)), np.array(his_val_loss))
        np.save(path.join(base_path_his, "Acc_Val_Branch_{}".format(branch_idx)), np.array(his_val_acc))

    def infer(self, input_batch: Union[Image, List[Image], torch.Tensor]):
        if not isinstance(input_batch, (Image, list)):
            input_batch = Image(input_batch)
        if type(input_batch) is Image:
            input_batch = [input_batch]
        if type(input_batch) is list:
            input_batch = torch.stack([torch.tensor(v.data) for v in input_batch])
        print('input_batch shape', input_batch.shape)
        cpu_device = torch.device('cpu')

        input_batch = input_batch.to(device=self.device, dtype=torch.float)
        self.model.eval()
        out_emotions, out_va = self.model(input_batch)

        # categorical result
        softmax_ = nn.Softmax(dim=0)
        categorical_results = out_emotions[:self.ensemble_size]  # a list of #n torch tensors where n is number of branches
        overall_emotion_preds = torch.zeros(categorical_results[0].size()).to(self.device) # size: batchsize * 8
        for o_eval, outputs_per_branch_eval in enumerate(categorical_results, 0):
            _, preds_indices = torch.max(outputs_per_branch_eval, 1)
            for v_i, v_p in enumerate(preds_indices, 0):
                overall_emotion_preds[v_i, v_p] += 1
        ensemble_emotion_results = [Category(prediction=int(o.argmax(dim=0)), confidence=float(max(softmax_(o))),
                                    description=datasets.AffectNetCategorical.get_class(int(o.argmax(dim=0))))
                                    for o in overall_emotion_preds]
        # dimension result
        dimensional_results = out_va[:self.ensemble_size]
        overall_dimension_preds = torch.zeros(dimensional_results[0].size()).to(cpu_device)
        for evaluate_branch in range(self.ensemble_size):
            out_va_eval_cpu = out_va[evaluate_branch].detach().to(cpu_device)
            overall_dimension_preds += out_va_eval_cpu
        ensemble_dimension_results = overall_dimension_preds / self.ensemble_size

        return ensemble_emotion_results, ensemble_dimension_results

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
            self.__convert_to_onnx(path.join(self.temp_path, self.name_experiment,
                                             "onnx_model.onnx"), do_constant_folding)
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(path.join(self.temp_path, self.name_experiment), exist_ok=True)
            self.__convert_to_onnx(path.join(self.temp_path, self.name_experiment, "onnx_model.onnx"),
                                   do_constant_folding, verbose=False)

        self.__load_from_onnx(path.join(self.temp_path, self.name_experiment, "onnx_model.onnx"))

    def __convert_to_onnx(self, output_name, do_constant_folding=False, verbose=False):
        """
        Converts the loaded regular PyTorch model to an ONNX model and saves it to disk.
        :param output_name: path and name to save the model, e.g. "/models/onnx_model.onnx"
        :type output_name: str
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        # Input to the model #### edit the next 3 lines for onnx_input
        onnx_input = torch.randn(self.batch_size, 3, 96, 96)
        if self.device == "cuda":
            onnx_input = Variable(onnx_input.float().cuda(self.output_device), requires_grad=False)
        else:
            onnx_input = Variable(onnx_input.float(), requires_grad=False)
        # Export the model
        torch.onnx.export(self.model,
                          onnx_input,
                          output_name,
                          verbose=verbose,
                          opset_version=11,
                          enable_onnx_checker=True,
                          do_constant_folding=do_constant_folding,
                          input_names=['onnx_input'],
                          output_names=['onnx_output'],
                          dynamic_axes={'onnx_input': {0: 'n'},
                                        'onnx_output': {0: 'n'}})

    def __load_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.
        :param path: path to ONNX model
        :type path: str
        """
        self.ort_session = onnxruntime.InferenceSession(path)

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError
    # def download(self):



