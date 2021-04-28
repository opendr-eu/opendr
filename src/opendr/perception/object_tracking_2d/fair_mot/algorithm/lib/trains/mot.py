from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.losses import FocalLoss
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.losses import (
    RegL1Loss,
    RegLoss,
    NormRegL1Loss,
    RegWeightedL1Loss,
)
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.decode import (
    mot_decode,
)
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.utils import (
    _sigmoid,
    _tranpose_and_gather_feat,
)
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.utils.post_process import (
    ctdet_post_process,
)
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(
        self,
        mse_loss,
        reg_loss,
        dense_wh,
        cat_spec_wh,
        reid_dim,
        nID,
        norm_wh,
        num_stacks,
        wh_weight,
        off_weight,
        id_weight,
        reg_offset,
        hm_weight,
    ):

        self.mse_loss = mse_loss
        self.reg_loss = reg_loss
        self.dense_wh = dense_wh
        self.cat_spec_wh = cat_spec_wh
        self.reid_dim = reid_dim
        self.nID = nID
        self.norm_wh = norm_wh
        self.num_stacks = num_stacks
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.id_weight = id_weight
        self.reg_offset = reg_offset
        self.hm_weight = hm_weight

        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if mse_loss else FocalLoss()
        self.crit_reg = (
            RegL1Loss()
            if reg_loss == "l1"
            else RegLoss()
            if reg_loss == "sl1"
            else None
        )
        self.crit_wh = (
            torch.nn.L1Loss(reduction="sum")
            if dense_wh
            else NormRegL1Loss()
            if norm_wh
            else RegWeightedL1Loss()
            if cat_spec_wh
            else self.crit_reg
        )
        self.emb_dim = reid_dim
        self.nID = nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(self.num_stacks):
            output = outputs[s]
            if not self.mse_loss:
                output["hm"] = _sigmoid(output["hm"])

            hm_loss += self.crit(output["hm"], batch["hm"]) / self.num_stacks
            if self.wh_weight > 0:
                wh_loss += (
                    self.crit_reg(
                        output["wh"], batch["reg_mask"], batch["ind"], batch["wh"],
                    )
                    / self.num_stacks
                )

            if self.reg_offset and self.off_weight > 0:
                off_loss += (
                    self.crit_reg(
                        output["reg"], batch["reg_mask"], batch["ind"], batch["reg"],
                    )
                    / self.num_stacks
                )

            if self.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output["id"], batch["ind"])
                id_head = id_head[batch["reg_mask"] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch["ids"][batch["reg_mask"] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
        )

        loss = (
            torch.exp(-self.s_det) * det_loss
            + torch.exp(-self.s_id) * id_loss
            + (self.s_det + self.s_id)
        )
        loss *= 0.5

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "id_loss": id_loss,
        }
        return loss, loss_stats


class MotTrainer(BaseTrainer):
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
        super(MotTrainer, self).__init__(
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
        )

    def _get_losses(
        self,
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
        loss_states = ["loss", "hm_loss", "wh_loss", "off_loss", "id_loss"]
        loss = MotLoss(
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
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output["reg"] if self.opt.reg_offset else None
        dets = mot_decode(
            output["hm"],
            output["wh"],
            reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh,
            K=self.opt.K,
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(),
            batch["meta"]["c"].cpu().numpy(),
            batch["meta"]["s"].cpu().numpy(),
            output["hm"].shape[2],
            output["hm"].shape[3],
            output["hm"].shape[1],
        )
        results[batch["meta"]["img_id"].cpu().numpy()[0]] = dets_out[0]
