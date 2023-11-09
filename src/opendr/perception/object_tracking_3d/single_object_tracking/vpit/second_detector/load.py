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
from torch.nn import functional

from google.protobuf import text_format

from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.protos import (
    pipeline_pb2,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.builder import (
    target_assigner_builder,
    voxel_builder,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.pytorch.builder import (
    box_coder_builder,
    lr_scheduler_builder,
    optimizer_builder,
    second_builder,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.torchplus_tanet.train import (
    MixedPrecisionWrapper,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.siamese import SiameseConvNet


class BCEWeightedLoss(nn.Module):
    def __init__(self, logits=True):
        super(BCEWeightedLoss, self).__init__()
        self.logits = logits

    def forward(self, input, target, weight=None):
        if self.logits:
            return functional.binary_cross_entropy_with_logits(
                input, target, weight, size_average=True
            )
        else:
            return functional.binary_cross_entropy(
                input, target, weight, size_average=True
            )


class SoftMarginLoss(nn.Module):
    def __init__(self):
        super(SoftMarginLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return functional.soft_margin_loss(input, target, reduction='elementwise_mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets, weights=None):
        if self.logits:
            inputs = torch.sigmoid(inputs)
        pt = torch.ones_like(inputs)
        pt[targets == 0] = (1 - inputs[targets == 0])
        pt[targets == 1] = inputs[targets == 1]
        if self.alpha:
            at = torch.ones_like(inputs)
            at[targets == 0] = (1 - self.alpha)
            at[targets == 1] = self.alpha

        loss = functional.binary_cross_entropy(inputs, targets, weight=weights, reduce=False)
        loss *= (1 - pt) ** self.gamma
        if self.alpha:
            loss *= at
        return loss.mean()


def create_model(
    config_path,
    device,
    optimizer_name,
    optimizer_params,
    lr,
    lr_schedule_name,
    lr_schedule_params,
    feature_blocks,
    loss_function,
    log=print,
    verbose=False,
    bof_mode="none",
    overwrite_strides=None,
    upscaling_mode="none",
):

    loss_scale = None

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(
        target_assigner_cfg, bv_range, box_coder
    )
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, device, bof_mode,
        overwrite_strides=overwrite_strides, upscaling_mode=upscaling_mode, feature_blocks=feature_blocks
    )
    net.device = device
    net.bv_range = bv_range
    net.point_cloud_range = voxel_generator.point_cloud_range
    net.voxel_size = model_cfg.voxel_generator.voxel_size

    if loss_function == "bce":
        net.criterion = BCEWeightedLoss()
    elif loss_function == "focal":
        net.criterion = FocalLoss(alpha=0, gamma=1)  # .to(self.device)
    elif loss_function == "softmargin":
        net.criterion = SoftMarginLoss()
    net.feature_blocks = feature_blocks

    model = SiameseConvNet(net)
    model.to(device)

    if verbose:
        log("num_trainable parameters:", len(list(net.parameters())))
        for n, p in net.named_parameters():
            log(n, p.shape)
    ######################
    # BUILD OPTIMIZER
    ######################
    gstep = net.get_global_step() - 1
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build_online(
        optimizer_name, optimizer_params, lr, model.parameters()
    )
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = MixedPrecisionWrapper(optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    lr_scheduler = lr_scheduler_builder.build_online(
        lr_schedule_name, lr_schedule_params, mixed_optimizer, gstep
    )
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    return (
        model,
        input_cfg,
        train_cfg,
        eval_input_cfg,
        model_cfg,
        voxel_generator,
        target_assigner,
        mixed_optimizer,
        lr_scheduler,
        float_dtype,
        loss_scale,
        class_names,
        center_limit_range,
    )


def load_from_checkpoint(
    model,
    mixed_optimizer,
    path,
    lr_schedule_name,
    lr_schedule_params,
    device=None,
    load_optimizer=True
):
    all_params = torch.load(path, map_location=device)
    model.load_state_dict(all_params["siamese_model"], strict=False)
    if load_optimizer:
        mixed_optimizer.load_state_dict(all_params["optimizer"])
    gstep = model.branch.get_global_step() - 1
    lr_scheduler = lr_scheduler_builder.build_online(
        lr_schedule_name, lr_schedule_params, mixed_optimizer, gstep if load_optimizer else -1
    )
    return lr_scheduler
