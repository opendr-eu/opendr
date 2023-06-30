'''
This code is based on https://github.com/thuiar/MIntRec under MIT license:

MIT License

Copyright (c) 2022 Hanlei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''


import torch
import torch.nn.functional as F
import logging
from torch import nn
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.utils.functions import (
    restore_model,
    save_model
)
from tqdm import trange, tqdm
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.utils.metrics import AverageMeter, Metrics

import shutil
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
__all__ = ['MULT']


class MULT:

    def __init__(self, args, model):

        self.logger = logging.getLogger(args.logger_name)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, verbose=True, patience=args.wait_patience)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.audio_distill_criterion = nn.KLDivLoss(reduction='batchmean')
        self.video_distill_criterion = nn.KLDivLoss(reduction='batchmean')
        self.language_distill_criterion = nn.KLDivLoss(reduction='batchmean')

        self.attention_audio_distill_criterion = nn.KLDivLoss(reduction='batchmean')
        self.attention_video_distill_criterion = nn.KLDivLoss(reduction='batchmean')
        self.attention_language_distill_criterion = nn.KLDivLoss(reduction='batchmean')

        self.criterion_audio = nn.CrossEntropyLoss()
        self.criterion_video = nn.CrossEntropyLoss()
        self.criterion_language = nn.CrossEntropyLoss()

        self.metrics = Metrics(args)
        self.best_eval_score = 0

    def train_joint(self, train_dataloader, val_dataloader):
        best_audio = 0
        best_video = 0
        best_language = 0
        best_joint = 0

        silent = self.logger.getEffectiveLevel() >= 40

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch", disable=silent):
            self.model.train()
            loss_record = AverageMeter()
            loss_record_audio = AverageMeter()
            loss_record_video = AverageMeter()
            loss_record_language = AverageMeter()

            loss_record_audio_distill = AverageMeter()
            loss_record_video_distill = AverageMeter()
            loss_record_language_distill = AverageMeter()
            T = self.args.t
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=silent)):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    preds, audio_preds, video_preds, language_preds, attention_audio, attention_video, \
                            attention_language, attention_audio_dist, attention_video_dist, attention_language_dist = \
                            self.model(text_feats, video_feats, audio_feats)

                    loss = self.criterion(preds, label_ids)
                    if self.args.attentiondist:
                        loss_distill_audio = self.attention_audio_distill_criterion(
                            F.log_softmax(
                                attention_audio_dist.reshape(
                                    attention_audio_dist.shape[0] * attention_audio_dist.shape[1], -1) / T, dim=1), F.softmax(
                                attention_audio.reshape(
                                    attention_audio.shape[0] * attention_audio.shape[1], -1).detach() / T, dim=1)) * T * T
                        loss_distill_video = self.attention_video_distill_criterion(
                            F.log_softmax(
                                attention_video_dist.reshape(
                                    attention_video_dist.shape[0] * attention_video_dist.shape[1], -1) / T, dim=1), F.softmax(
                                attention_video.reshape(
                                    attention_video.shape[0] * attention_video.shape[1], -1).detach() / T, dim=1)) * T * T
                        loss_distill_language = self.attention_language_distill_criterion(
                            F.log_softmax(
                                attention_language_dist.reshape(
                                    attention_language_dist.shape[0] * attention_language_dist.shape[1], -1) / T, dim=1),
                            F.softmax(attention_language.reshape(
                                attention_language.shape[0] * attention_language.shape[1], -1).detach() / T, dim=1)) * T * T
                    elif self.args.attentiondistboth:
                        loss_distill_audio1 = self.attention_audio_distill_criterion(
                            F.log_softmax(
                                attention_audio_dist.reshape(
                                    attention_audio_dist.shape[0] * attention_audio_dist.shape[1], -1) / T, dim=1), F.softmax(
                                attention_audio.reshape(
                                    attention_audio.shape[0] * attention_audio.shape[1], -1).detach() / T, dim=1)) * T * T
                        loss_distill_video1 = self.attention_video_distill_criterion(
                            F.log_softmax(
                                attention_video_dist.reshape(
                                    attention_video_dist.shape[0] * attention_video_dist.shape[1], -1) / T, dim=1), F.softmax(
                                attention_video.reshape(
                                    attention_video.shape[0] * attention_video.shape[1], -1).detach() / T, dim=1)) * T * T
                        loss_distill_language1 = self.attention_language_distill_criterion(
                            F.log_softmax(
                                attention_language_dist.reshape(attention_language_dist.shape[0] *
                                                                attention_language_dist.shape[1], -1) / T, dim=1),
                            F.softmax(attention_language.reshape(attention_language.shape[0] *
                                                                 attention_language.shape[1], -1).detach() / T, dim=1)) * T * T
                        loss_distill_audio2 = self.audio_distill_criterion(F.log_softmax(
                            audio_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T
                        loss_distill_video2 = self.video_distill_criterion(F.log_softmax(
                            video_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T
                        loss_distill_language2 = self.language_distill_criterion(F.log_softmax(
                            language_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T
                        loss_distill_audio = loss_distill_audio1 + loss_distill_audio2
                        loss_distill_video = loss_distill_video1 + loss_distill_video2
                        loss_distill_language = loss_distill_language1 + loss_distill_language2
                    else:
                        loss_distill_audio = self.audio_distill_criterion(F.log_softmax(
                            audio_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T
                        loss_distill_video = self.video_distill_criterion(F.log_softmax(
                            video_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T
                        loss_distill_language = self.language_distill_criterion(F.log_softmax(
                            language_preds / T, dim=1), F.softmax(preds.detach() / T, dim=1)) * T * T

                    loss_audio = self.criterion_audio(audio_preds, label_ids)
                    loss_video = self.criterion_video(video_preds, label_ids)
                    loss_language = self.criterion_language(language_preds, label_ids)

                    total_loss = self.args.gamma * loss + self.args.alpha * loss_distill_audio + \
                        self.args.alpha * loss_distill_video + self.args.alpha * \
                        loss_distill_language + self.args.betha * loss_audio + \
                        self.args.betha * loss_video + self.args.betha * loss_language

                    self.optimizer.zero_grad()

                    total_loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    loss_record_audio.update(loss_audio.item(), label_ids.size(0))
                    loss_record_video.update(loss_video.item(), label_ids.size(0))
                    loss_record_language.update(loss_language.item(), label_ids.size(0))

                    loss_record_audio_distill.update(loss_distill_audio.item(), label_ids.size(0))
                    loss_record_video_distill.update(loss_distill_video.item(), label_ids.size(0))
                    loss_record_language_distill.update(loss_distill_language.item(), label_ids.size(0))

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters()
                                                  if param.requires_grad], self.args.grad_clip)

                    self.optimizer.step()

            outputs, outputs_audio, outputs_video, outputs_language = self._get_outputs_joint(val_dataloader, silent=silent)
            self.scheduler.step(outputs['loss'] + outputs_audio['loss'] + outputs_video['loss'] + outputs_language['loss'])
            eval_score = outputs[self.args.eval_monitor]

            eval_score_audio = outputs_audio[self.args.eval_monitor]
            eval_score_video = outputs_video[self.args.eval_monitor]
            eval_score_language = outputs_language[self.args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'train_loss_audio': round(loss_record_audio.avg, 4),
                'train_loss_video': round(loss_record_video.avg, 4),
                'train_loss_language': round(loss_record_language.avg, 4),
                'eval_score': round(eval_score, 4),
                'eval_score_audio': round(eval_score_audio, 4),
                'eval_score_video': round(eval_score_video, 4),
                'eval_score_language': round(eval_score_language, 4),
            }

            self.logger.debug("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.debug("  %s = %s", key, str(eval_results[key]))
            scores = {
                'audio': eval_score_audio,
                'video': eval_score_video,
                'language': eval_score_language,
                'joint': eval_score}
            save_model(self.model, self.args.model_output_path, scores, name='last.pth')
            if best_audio < eval_score_audio:
                best_audio = eval_score_audio
                shutil.copyfile(
                    os.path.join(
                        self.args.model_output_path, 'last.pth'), os.path.join(
                        self.args.model_output_path, 'best_audio.pth'))
            if best_video < eval_score_video:
                best_video = eval_score_video
                shutil.copyfile(
                    os.path.join(
                        self.args.model_output_path, 'last.pth'), os.path.join(
                        self.args.model_output_path, 'best_video.pth'))

            if best_language < eval_score_language:
                best_language = eval_score_language
                shutil.copyfile(
                    os.path.join(
                        self.args.model_output_path, 'last.pth'), os.path.join(
                        self.args.model_output_path, 'best_language.pth'))
            if best_joint < eval_score:
                best_joint = eval_score
                shutil.copyfile(
                    os.path.join(
                        self.args.model_output_path, 'last.pth'), os.path.join(
                        self.args.model_output_path, 'best_joint.pth'))

    def _get_outputs_joint(self, dataloader, return_sample_results=False, show_results=False, silent=False):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.args.num_labels)).to(self.device)
        total_logits_audio = torch.empty((0, self.args.num_labels)).to(self.device)
        total_logits_video = torch.empty((0, self.args.num_labels)).to(self.device)
        total_logits_language = torch.empty((0, self.args.num_labels)).to(self.device)

        loss_record = AverageMeter()
        loss_record_audio = AverageMeter()
        loss_record_video = AverageMeter()
        loss_record_language = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration", disable=silent):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):

                logits, audio_logits, video_logits, language_logits, _, _, _, _, _, _ = self.model(
                    text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_logits_audio = torch.cat((total_logits_audio, audio_logits))
                total_logits_video = torch.cat((total_logits_video, video_logits))
                total_logits_language = torch.cat((total_logits_language, language_logits))

                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

                loss_audio = self.criterion(audio_logits, label_ids)
                loss_record_audio.update(loss_audio.item(), label_ids.size(0))

                loss_video = self.criterion(video_logits, label_ids)
                loss_record_video.update(loss_video.item(), label_ids.size(0))

                loss_language = self.criterion(language_logits, label_ids)
                loss_record_language.update(loss_language.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        total_probs_audio = F.softmax(total_logits_audio.detach(), dim=1)
        total_maxprobs_audio, total_preds_audio = total_probs_audio.max(dim=1)

        y_pred_audio = total_preds_audio.cpu().numpy()

        outputs_audio = self.metrics(y_true, y_pred_audio, show_results=show_results)
        outputs_audio.update({'loss': loss_record_audio.avg})

        total_probs_video = F.softmax(total_logits_video.detach(), dim=1)
        total_maxprobs_video, total_preds_video = total_probs_video.max(dim=1)

        y_pred_video = total_preds_video.cpu().numpy()

        outputs_video = self.metrics(y_true, y_pred_video, show_results=show_results)
        outputs_video.update({'loss': loss_record_video.avg})

        total_probs_language = F.softmax(total_logits_language.detach(), dim=1)
        total_maxprobs_language, total_preds_language = total_probs_language.max(dim=1)

        y_pred_language = total_preds_language.cpu().numpy()

        outputs_language = self.metrics(y_true, y_pred_language, show_results=show_results)
        outputs_language.update({'loss': loss_record_language.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )
            outputs_audio.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred_audio
                }
            )
            outputs_video.update(

                {
                    'y_true': y_true,
                    'y_pred': y_pred_video
                }
            )
            outputs_language.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred_language
                }
            )
        return outputs, outputs_audio, outputs_video, outputs_language

    def train(self, train_dataloader, val_dataloader):
        if self.args.mode == 'joint':
            return self.train_joint(train_dataloader, val_dataloader)

        silent = self.logger.getEffectiveLevel() >= 40
        best_joint = 0
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch", disable=silent):
            self.model.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=silent)):
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    preds, last_hiddens = self.model(text_feats, video_feats, audio_feats)

                    loss = self.criterion(preds, label_ids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters()
                                                  if param.requires_grad], self.args.grad_clip)

                    self.optimizer.step()

            outputs = self._get_outputs(val_dataloader, silent=silent)
            self.scheduler.step(outputs['loss'])
            eval_score = outputs[self.args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.debug("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.debug("  %s = %s", key, str(eval_results[key]))

            scores = {'joint': eval_score}
            save_model(self.model, self.args.model_output_path, scores, name='last.pth')
            if best_joint < eval_score:
                best_joint = eval_score
                shutil.copyfile(
                    os.path.join(
                        self.args.model_output_path, 'last.pth'), os.path.join(
                        self.args.model_output_path, 'best.pth'))

    def _get_outputs(self, dataloader, return_sample_results=False, show_results=False, silent=False):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.args.num_labels)).to(self.device)

        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration", disable=silent):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):

                logits, last_hiddens = self.model(text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    def infer(self, text_feats, modality, audio_feats=None, video_feats=None):
        if audio_feats is None:
            audio_feats = torch.zeros(
                text_feats.size(0),
                self.args.max_seq_length_audio,
                self.args.audio_feat_dim).to(
                self.args.device)
        if video_feats is None:
            video_feats = torch.zeros(
                text_feats.size(0),
                self.args.max_seq_length_video,
                self.args.video_feat_dim).to(
                self.args.device)
        self.model.eval()
        with torch.no_grad():
            if self.args.mode == 'joint':
                logits, logits_audio, logits_video, logits_language, _, _, _, _, _, _ = self.model.model(
                    text_feats, video_feats, audio_feats)
                if modality == 'audio':
                    logits = logits_audio
                elif modality == 'video':
                    logits = logits_video
                elif modality == 'language':
                    logits = logits_language
            else:
                logits, _ = self.model.model(text_feats, video_feats, audio_feats)

        probs = F.softmax(logits.detach(), dim=1)

        maxprobs, preds = probs.mean(dim=0).max(dim=0)
        y_pred = preds.cpu().numpy()
        maxprob = maxprobs.cpu().numpy()
        return (y_pred, maxprob.item())
        # return (labels[y_pred], maxprob.item())

    def test(self, dataloader, modality='joint', restore_best_model=True):
        silent = self.logger.getEffectiveLevel() >= 40
        verbose = self.logger.getEffectiveLevel() <= 10

        if self.args.mode == 'joint':
            if restore_best_model:
                self.model = restore_model(self.model, self.args.model_output_path, name='best_{}.pth'.format(modality))
            test_results, test_results_audio, test_results_video, test_results_language = self._get_outputs_joint(
                dataloader, return_sample_results=True, show_results=verbose, silent=silent)

            if modality == 'audio':
                return test_results_audio
            elif modality == 'video':
                return test_results_video
            elif modality == 'language':
                return test_results_language
        else:
            if restore_best_model:
                self.model = restore_model(self.model, self.args.model_output_path, name='best.pth')
            test_results = self._get_outputs(dataloader, return_sample_results=True, show_results=verbose, silent=silent)

        return test_results
