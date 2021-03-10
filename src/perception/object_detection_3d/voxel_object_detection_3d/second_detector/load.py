import torch

import pathlib
from google.protobuf import text_format
import shutil

from perception.object_detection_3d.voxel_object_detection_3d.second_detector.protos import (
    pipeline_pb2,
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.builder import (
    target_assigner_builder,
    voxel_builder,
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.builder import (
    box_coder_builder,
    lr_scheduler_builder,
    optimizer_builder,
    second_builder,
)
import torchplus


def create_model(
    config_path, device, optimizer_name,
    optimizer_params, lr, lr_schedule_name, lr_schedule_params,
    log=print
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
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, device)
    net.to(device)
    net.device = device
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
    optimizer = optimizer_builder.build_online(optimizer_name, optimizer_params, lr, net.parameters())
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    lr_scheduler = lr_scheduler_builder.build_online(lr_schedule_name, lr_schedule_params, optimizer, gstep)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    return (
        net,
        input_cfg,
        train_cfg,
        eval_input_cfg,
        model_cfg,
        train_cfg,
        voxel_generator,
        target_assigner,
        mixed_optimizer,
        lr_scheduler,
        float_dtype,
        loss_scale,
        class_names,
        center_limit_range,
    )


def load(
    model_dir, config_path, device, optimizer_name,
    optimizer_params, lr, lr_schedule_name, lr_schedule_params,
    create_folder=False, result_path=None, log=print
):

    loss_scale = None

    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / "results"
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
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
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, device=device)
    net.to(device)
    net.device = device
    log("num_trainable parameters:", len(list(net.parameters())))
    for n, p in net.named_parameters():
        log(n, p.shape)
    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net], device=device)
    gstep = net.get_global_step() - 1
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build_online(optimizer_name, optimizer_params, lr, net.parameters())
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir, [mixed_optimizer], device=device)
    lr_scheduler = lr_scheduler_builder.build_online(lr_schedule_name, lr_schedule_params, optimizer, gstep)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    return (
        net,
        input_cfg,
        train_cfg,
        eval_input_cfg,
        model_cfg,
        train_cfg,
        voxel_generator,
        target_assigner,
        mixed_optimizer,
        lr_scheduler,
        model_dir,
        float_dtype,
        loss_scale,
        result_path,
        class_names,
        center_limit_range,
    )
