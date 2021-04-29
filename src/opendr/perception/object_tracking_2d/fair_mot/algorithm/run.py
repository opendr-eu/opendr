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

import os
import torch
import numpy as np
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.trains.mot import MotTrainer
from opendr.perception.object_tracking_2d.logger import Logger
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.tracking_utils.evaluation import Evaluator


def train(
    model,
    infer,
    optimizer,
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    gpus,
    chunk_sizes,
    num_iters,
    exp_id,
    device,
    hide_data_time,
    print_iter,
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
    num_epochs,
    lr_step,
    save_dir,
    lr,
    reg_offset,
    hm_weight,
    checkpoints_path,
    checkpoint_after_iter,
    start_epoch,
    val_epochs=5,
    log=print,
):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trainer = MotTrainer(
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
    trainer.set_device(gpus, chunk_sizes, device)

    def save(iter):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch,
                'state_dict': state_dict}
        if not (optimizer is None):
            data['optimizer'] = optimizer.state_dict()
        torch.save(data, os.path.join(
            checkpoints_path,
            f"checkpoint_{iter}.pth"
        ))

    last_eval_result = {}

    for epoch in range(start_epoch + 1, num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, train_loader, checkpoint_after_iter, save, log)
        log(Logger.LOG_WHEN_NORMAL, "epoch: {} |".format(epoch))

        if val_epochs > 0:
            if epoch % val_epochs == 0:
                last_eval_result = evaluate(infer, val_dataset, "evaluation " + str(epoch))
                log(Logger.LOG_WHEN_NORMAL, last_eval_result)
        if epoch in lr_step:
            lr = lr * (0.1 ** (lr_step.index(epoch) + 1))
            log(Logger.LOG_WHEN_NORMAL, "Drop LR to", lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    print()
    return last_eval_result


def evaluate(
    infer,
    val_dataset,
    name="default",
):

    gt_targets = []
    output_targets = []
    evaluator = Evaluator()

    frame_id = 0

    for frame_id in range(len(val_dataset)):
        image, target = val_dataset[frame_id]
        output_target = infer(image, frame_id)
        output_targets.append(output_target)
        gt_targets.append(target)

    gt_tlwhs = [
        [
            np.array([t.top, t.left, t.width, t.height])
            for t in ts.data
        ] for ts in gt_targets
    ]
    output_tlwhs = [
        [
            np.array([t.top, t.left, t.width, t.height])
            for t in ts.data
        ] for ts in output_targets
    ]

    gt_ids = [
        [
            t.id for t in ts.data
        ] for ts in gt_targets
    ]
    output_ids = [
        [
            t.id for t in ts.data
        ] for ts in output_targets
    ]

    acc = evaluator.eval_all(
        gt_tlwhs,
        gt_ids,
        output_tlwhs,
        output_ids,
    )

    summary = Evaluator.get_summary([acc], [name])

    return summary
