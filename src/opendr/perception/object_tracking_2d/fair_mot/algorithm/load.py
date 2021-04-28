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

import torch
from opendr.perception.object_tracking_2d.logger import Logger


def load_from_checkpoint(
    model, model_path, optimizer, resume=False, lr=None, lr_step=None, log=print
):
    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage
    )
    log(Logger.LOG_WHEN_NORMAL, "loaded {}, epoch {}".format(model_path, checkpoint["epoch"]))
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = (
        "If you see this, your model does not fully load the " +
        "pre-trained weight. Please make sure " +
        "you have correctly specified --arch xxx " +
        "or set the correct --num_classes for your own dataset."
    )
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                log(
                    Logger.LOG_WHEN_NORMAL,
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}. {}".format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            log(Logger.LOG_WHEN_NORMAL, "Drop parameter {}.".format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            log(Logger.LOG_WHEN_NORMAL, "No param {}.".format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            log(Logger.LOG_WHEN_NORMAL, "Resumed optimizer with start lr", start_lr)
        else:
            log(Logger.LOG_WHEN_NORMAL, "No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
