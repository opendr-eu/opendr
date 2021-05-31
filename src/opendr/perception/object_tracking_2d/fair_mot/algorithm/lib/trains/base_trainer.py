from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.data_parallel import (
    DataParallel,
)
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.utils.utils import (
    AverageMeter,
)
from opendr.perception.object_tracking_2d.logger import Logger


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):

        if self.model.ort_session is None:
            outputs = self.model(batch["input"])
        else:
            outputs_flat = self.rpn_ort_session.run(None, {'data': np.array(batch["input"].cpu())})
            outputs = {}

            for i in range(len(self.model.heads_names)):
                outputs[self.model.heads_names[i]] = outputs_flat[i]

        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
        self,
        model,
        gpus,
        num_iters,
        exp_id,
        device,
        hide_data_time,
        print_iter,
        optimizer,
        mse_loss,
        reg_loss,
        dense_wh,
        cat_spec_wh,
        reid_dim,
        nID,
        norm_wh,
        num_stack,
        wh_weight,
        off_weight,
        id_weight,
        reg_offset,
        hm_weight,
    ):
        self.gpus = gpus
        self.num_iters = num_iters
        self.exp_id = exp_id
        self.device = device
        self.hide_data_time = hide_data_time
        self.print_iter = print_iter

        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(
            mse_loss,
            reg_loss,
            dense_wh,
            cat_spec_wh,
            reid_dim,
            nID,
            norm_wh,
            num_stack,
            wh_weight,
            off_weight,
            id_weight,
            reg_offset,
            hm_weight,
        )
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.optimizer.add_param_group({"params": self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus, chunk_sizes=chunk_sizes
            ).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader, save_iter, save, log=print):
        model_with_loss = self.model_with_loss
        if phase == "train":
            model_with_loss.train()
        else:
            if len(self.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if self.num_iters < 0 else self.num_iters
        bar = Bar("{}/{}".format("mot", self.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != "meta":
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = "{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} ".format(
                epoch,
                iter_id,
                num_iters,
                phase=phase,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch["input"].size(0)
                )
                Bar.suffix = Bar.suffix + "|{} {:.4f} ".format(l, avg_loss_stats[l].avg)
            if not self.hide_data_time:
                Bar.suffix = (
                    Bar.suffix + "|Data {dt.val:.3f}s({dt.avg:.3f}s) "
                    "|Net {bt.avg:.3f}s".format(dt=data_time, bt=batch_time)
                )
            if self.print_iter > 0:
                if iter_id % self.print_iter == 0:
                    log(
                        Logger.LOG_WHEN_NORMAL,
                        "{}/{}| {}".format("mot", self.exp_id, Bar.suffix),
                    )

            if save_iter > 0:
                if iter_id % save_iter == 0:
                    save(iter_id + num_iters * epoch)
                    log(
                        Logger.LOG_WHEN_NORMAL,
                        "Model saved",
                    )

            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret["time"] = bar.elapsed_td.total_seconds() / 60.0
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch("val", epoch, data_loader, -1, None)

    def train(self, epoch, data_loader, save_iter, save, log=print):
        return self.run_epoch("train", epoch, data_loader, save_iter, save, log)
