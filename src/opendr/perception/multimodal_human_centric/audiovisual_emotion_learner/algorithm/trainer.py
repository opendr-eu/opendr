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

from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.utils import (
     adjust_learning_rate,
     Logger,
     save_checkpoint,
     calculate_accuracy,
     AverageMeter
     )
import os
import torch.nn as nn
from torch import optim
import torch


def update_tensorboard(tensorboard_logger, logger_prefix, name, value, epoch_idx):
    tensorboard_logger.add_scalar(tag='{}/{}'.format(logger_prefix, name),
                                  scalar_value=value,
                                  global_step=epoch_idx + 1)
    tensorboard_logger.flush()


def train(model, train_loader, val_loader, learning_rate, momentum, dampening, weight_decay, n_epochs, save_dir, lr_steps,
          mod_drop='zerodrop', device='cpu', silent=False, verbose=True,
          tensorboard_logger=None, eval_mode='audiovisual', restore_best=False):

    metrics = {'train_loss': [], 'train_acc': []}
    train_logger = Logger(os.path.join(save_dir, 'train.log'),
                          ['epoch', 'loss', 'acc'])

    if val_loader is not None:
        metrics.update({'val_loss': [], 'val_acc': []})
        val_logger = Logger(os.path.join(save_dir, 'val.log'), ['epoch', 'loss', 'acc'])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay)

    best_acc = -1
    is_best = False
    for i in range(n_epochs):
        adjust_learning_rate(optimizer, i, learning_rate, lr_steps)

        train_loss, train_acc = train_one_epoch(i, train_loader, model, criterion, optimizer,
                                                train_logger, mod_drop, device, silent, verbose)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        if tensorboard_logger is not None:
            update_tensorboard(tensorboard_logger, 'train', 'loss', train_loss, i)
            update_tensorboard(tensorboard_logger, 'train', 'acc', train_acc, i)

        if val_loader is not None:
            validation_loss, validation_acc = val_one_epoch(
                i, val_loader, model, criterion, val_logger, eval_mode, device, silent, verbose)
            metrics['val_loss'].append(validation_loss)
            metrics['val_acc'].append(validation_acc)
            if tensorboard_logger is not None:
                update_tensorboard(tensorboard_logger, 'val', 'loss', validation_loss, i)
                update_tensorboard(tensorboard_logger, 'val', 'acc', validation_acc, i)

            is_best = validation_acc > best_acc
            if is_best and not silent and verbose:
                print('Validation accuracy improved from {} to {}.'.format(best_acc, validation_acc))

            best_acc = max(validation_acc, best_acc)
        else:
            is_best = train_acc > best_acc
            best_acc = max(train_acc, best_acc)

        state = {
            'epoch': i,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        save_checkpoint(state, is_best, save_dir, 'model')
    if restore_best:
        print('restoring best')
        if os.path.exists((os.path.join(save_dir, 'model_best.pth'))):
            print('restoring best')
            checkpoint = torch.load(os.path.join(save_dir, 'model_best.pth'))
            model.load_state_dict(checkpoint['state_dict'])
    return metrics


def apply_mask(audio_inputs, visual_inputs, targets, mod_drop):
    if mod_drop == 'zerodrop':
        coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1))/100
        vision_coefficients = 1 - coefficients
        coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
        vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(
            1, visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

        audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients,
                                 torch.zeros(audio_inputs.size()), audio_inputs), dim=0)
        visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients,
                                  visual_inputs, torch.zeros(visual_inputs.size())), dim=0)

        targets = torch.cat((targets, targets, targets, targets), dim=0)
        shuffle = torch.randperm(audio_inputs.size()[0])
        audio_inputs = audio_inputs[shuffle]
        visual_inputs = visual_inputs[shuffle]
        targets = targets[shuffle]
    elif mod_drop == 'noisedrop':
        audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)

        visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)

        targets = torch.cat((targets, targets, targets), dim=0)

        shuffle = torch.randperm(audio_inputs.size()[0])
        audio_inputs = audio_inputs[shuffle]
        visual_inputs = visual_inputs[shuffle]
        targets = targets[shuffle]

    return audio_inputs, visual_inputs, targets


def train_one_epoch(epoch, data_loader, model, criterion, optimizer, e_logger,
                    mod_drop='zerodrop', device='cpu', silent=False, verbose=True):
    model.train()

    loss = AverageMeter()
    acc = AverageMeter()

    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        targets = targets.to(device)
        with torch.no_grad():
            audio_inputs, visual_inputs, targets = apply_mask(audio_inputs, visual_inputs, targets, mod_drop)

        audio_inputs = audio_inputs.to(device)
        visual_inputs = visual_inputs.to(device)

        visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(
            visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2],
            visual_inputs.shape[3], visual_inputs.shape[4])

        outputs = model(audio_inputs, visual_inputs)
        loss_b = criterion(outputs, targets)

        loss.update(loss_b.data, audio_inputs.size(0))

        acc_1, _ = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))

        acc.update(acc_1, audio_inputs.size(0))

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()

        if not silent and (verbose or (not verbose and epoch % 10 == 0)):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.5f} ({acc.avg:.5f})\t'.format(
                      epoch,
                      i,
                      len(data_loader),
                      loss=loss,
                      acc=acc,
                  ))

    e_logger.log({
        'epoch': epoch,
        'loss': loss.avg.item(),
        'acc': acc.avg.item()
    })
    return loss.avg.item(), acc.avg.item()


def val_one_epoch(epoch, data_loader, model, criterion, logger=None,
                  mode='audiovisual', device='cpu', silent=False, verbose=True):
    model.eval()

    loss = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():
        for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):

            if mode == 'onlyaudio':
                inputs_visual = torch.zeros(inputs_visual.size())
            elif mode == 'noisyvideo':
                inputs_visual = torch.randn(inputs_visual.size())
            elif mode == 'onlyvideo':
                inputs_audio = torch.zeros(inputs_audio.size())
            elif mode == 'noisyaudio':
                inputs_audio = torch.randn(inputs_audio.size())

            inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
            inputs_visual = inputs_visual.reshape(
                inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2],
                inputs_visual.shape[3], inputs_visual.shape[4])

            targets = targets.to(device)
            inputs_audio = inputs_audio.to(device)
            inputs_visual = inputs_visual.to(device)

            outputs = model(inputs_audio, inputs_visual)
            loss_b = criterion(outputs, targets)

            acc1, _ = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            acc.update(acc1, inputs_audio.size(0))

            loss.update(loss_b.data, inputs_audio.size(0))
            if not silent and (verbose or (not verbose and epoch % 10 == 0)):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.5f} ({acc.avg:.5f})\t'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          loss=loss,
                          acc=acc))

    if logger is not None:
        logger.log({'epoch': epoch,
                    'loss': loss.avg.item(),
                    'acc': acc.avg.item()})

    return loss.avg.item(), acc.avg.item()
