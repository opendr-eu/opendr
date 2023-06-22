import math
from collections import OrderedDict

import continual as co
import torch
from continual import PaddingMode
from torch import nn

from .res import CoResStage, init_weights

from opendr.perception.activity_recognition.x3d.algorithm.operators import Swish
from .se import CoSe


def CoX3DTransform(
    dim_in: int,
    dim_out: int,
    temp_kernel_size: int,
    stride: int,
    dim_inner: int,
    num_groups: int,
    stride_1x1=False,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    dilation=1,
    norm_module=torch.nn.BatchNorm3d,
    se_ratio=0.0625,
    swish_inner=True,
    block_idx=0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",  # "frame" or "clip"
):
    """
    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temp_kernel_size (int): the temporal kernel sizes of the middle
            convolution in the bottleneck.
        stride (int): the stride of the bottleneck.
        dim_inner (int): the inner dimension of the block.
        num_groups (int): number of groups for the convolution. num_groups=1
            is for standard ResNet like networks, and num_groups>1 is for
            ResNeXt like networks.
        stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
            apply stride to the 3x3 conv.
        inplace_relu (bool): if True, calculate the relu on the original
            input without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dilation (int): size of dilation.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
        se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
            channel dimensionality being se_ratio times the Tx3x3 conv dim.
        swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
            apply ReLU to the Tx3x3 conv.
    """
    (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)

    a = co.Conv3d(
        dim_in,
        dim_inner,
        kernel_size=(1, 1, 1),
        stride=(1, str1x1, str1x1),
        padding=(0, 0, 0),
        bias=False,
    )

    a_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    a_relu = torch.nn.ReLU(inplace=inplace_relu)

    # Tx3x3, BN, ReLU.
    b = co.Conv3d(
        dim_inner,
        dim_inner,
        kernel_size=(temp_kernel_size, 3, 3),
        stride=(1, str3x3, str3x3),
        padding=(int(temp_kernel_size // 2), dilation, dilation),
        groups=num_groups,
        bias=False,
        dilation=(1, dilation, dilation),
        temporal_fill=temporal_fill,
    )

    b_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    # Apply SE attention or not
    use_se = True if (block_idx + 1) % 2 else False
    if se_ratio > 0.0 and use_se:
        se = CoSe(
            temporal_window_size,
            dim_in=dim_inner,
            ratio=se_ratio,
            temporal_fill=temporal_fill,
            scope=se_scope,
        )

    b_relu = co.forward_stepping(
        Swish()  # nn.SELU is the same as Swish
        if swish_inner
        else nn.ReLU(inplace=inplace_relu)
    )

    # 1x1x1, BN.
    c = co.Conv3d(
        dim_inner,
        dim_out,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        bias=False,
    )
    c_bn = co.forward_stepping(
        norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)
    )
    c_bn.transform_final_bn = True

    return co.Sequential(
        OrderedDict(
            [
                ("a", a),
                ("a_bn", a_bn),
                ("a_relu", a_relu),
                ("b", b),
                ("b_bn", b_bn),
                *([("se", se)] if use_se else []),
                ("b_relu", b_relu),
                ("c", c),
                ("c_bn", c_bn),
            ]
        )
    )


def CoX3DHead(
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    pool_size: int,
    dropout_rate=0.0,
    act_func="softmax",
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=torch.nn.BatchNorm3d,
    bn_lin5_on=False,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    no_pool=False,
):
    """
    Continual X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """
    modules = []
    modules.append(
        (
            "conv_5",
            co.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
        )
    )
    modules.append(
        ("conv_5_bn", norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt))
    )
    modules.append(("conv_5_relu", torch.nn.ReLU(inplace_relu)))

    if no_pool:
        return co.Sequential(OrderedDict(modules))

    avg_pool = co.Sequential(
        co.Lambda(lambda x: x.mean(dim=(-1, -2))),
        co.AvgPool1d(temporal_window_size, stride=1, temporal_fill=temporal_fill),
    )

    modules.append(("avg_pool", avg_pool))
    modules.append(
        (
            "lin_5",
            co.Conv1d(
                dim_inner,
                dim_out,
                kernel_size=(1,),
                stride=(1,),
                padding=(0,),
                bias=False,
            ),
        )
    )
    if bn_lin5_on:
        modules.append(
            ("lin_5_bn", torch.nn.BatchNorm1d(num_features=dim_out, eps=eps, momentum=bn_mmt))
        )

    modules.append(("lin_5_relu", torch.nn.ReLU(inplace_relu)))

    if dropout_rate > 0.0:
        modules.append(("dropout", torch.nn.Dropout(dropout_rate)))

    # Perform FC in a fully convolutional manner. The FC layer will be
    # initialized with a different std comparing to convolutional layers.
    modules.append(
        ("projection", co.Linear(dim_out, num_classes, bias=True, channel_dim=1))
    )

    def view(x):
        return x.view(x.shape[0], -1)

    modules.append(("view", co.Lambda(view, forward_only_fn=view)))

    return co.Sequential(OrderedDict(modules))


def CoX3DStem(
    dim_in: int,
    dim_out: int,
    kernel: int,
    stride: int,
    padding: int,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=torch.nn.BatchNorm3d,
    temporal_fill: PaddingMode = "zeros",
    *args,
    **kwargs,
):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu followed by a
    spatiotemporal pooling.

    Args:
        dim_in (int): the channel dimension of the input. Normally 3 is used
            for rgb input, and 2 or 3 is used for optical flow input.
        dim_out (int): the output dimension of the convolution in the stem
            layer.
        kernel (list): the kernel size of the convolution in the stem layer.
            temporal kernel size, height kernel size, width kernel size in
            order.
        stride (list): the stride size of the convolution in the stem layer.
            temporal kernel stride, height kernel size, width kernel size in
            order.
        padding (int): the padding size of the convolution in the stem
            layer, temporal padding size, height padding size, width
            padding size in order.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
    """
    conv_xy = co.Conv3d(
        dim_in,
        dim_out,
        kernel_size=(1, kernel[1], kernel[2]),
        stride=(1, stride[1], stride[2]),
        padding=(0, padding[1], padding[2]),
        bias=False,
    )

    conv = co.Conv3d(
        dim_out,
        dim_out,
        kernel_size=(kernel[0], 1, 1),
        stride=(stride[0], 1, 1),
        padding=(padding[0], 0, 0),
        bias=False,
        groups=dim_out,
        temporal_fill=temporal_fill,
    )

    bn = norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)

    relu = torch.nn.ReLU(inplace_relu)

    # Wrap in sequential to match weight specification
    return co.Sequential(
        OrderedDict(
            [
                (
                    "pathway0_stem",
                    co.Sequential(
                        OrderedDict(
                            [
                                ("conv_xy", conv_xy),
                                ("conv", conv),
                                ("bn", bn),
                                ("relu", relu),
                            ]
                        )
                    ),
                )
            ]
        )
    )


def CoX3D(
    dim_in: int,
    image_size: int,
    temporal_window_size: int,
    num_classes: int,
    x3d_conv1_dim: int,
    x3d_conv5_dim: int,
    x3d_num_groups: int,
    x3d_width_per_group: int,
    x3d_width_factor: float,
    x3d_depth_factor: float,
    x3d_bottleneck_factor: float,
    x3d_use_channelwise_3x3x3: bool,
    x3d_dropout_rate: float,
    x3d_head_activation: str,
    x3d_head_batchnorm: bool,
    x3d_fc_std_init: float,
    x3d_final_batchnorm_zero_init: bool,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",
    headless=False,
) -> co.Sequential:
    """
    Continual X3D model,
    adapted from https://github.com/facebookresearch/SlowFast

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """
    norm_module = torch.nn.BatchNorm3d
    exp_stage = 2.0
    dim_conv1 = x3d_conv1_dim

    num_groups = x3d_num_groups
    width_per_group = x3d_width_per_group
    dim_inner = num_groups * width_per_group

    w_mul = x3d_width_factor
    d_mul = x3d_depth_factor

    dim_res1 = _round_width(dim_conv1, w_mul)
    dim_res2 = dim_conv1
    dim_res3 = _round_width(dim_res2, exp_stage, divisor=8)
    dim_res4 = _round_width(dim_res3, exp_stage, divisor=8)
    dim_res5 = _round_width(dim_res4, exp_stage, divisor=8)

    block_basis = [
        # blocks, c, stride
        [1, dim_res2, 2],
        [2, dim_res3, 2],
        [5, dim_res4, 2],
        [3, dim_res5, 2],
    ]

    # Basis of temporal kernel sizes for each of the stage.
    temp_kernel = [
        [5],  # conv1 temporal kernels.
        [3],  # res2 temporal kernels.
        [3],  # res3 temporal kernels.
        [3],  # res4 temporal kernels.
        [3],  # res5 temporal kernels.
    ]

    modules = []

    s1 = CoX3DStem(
        dim_in=dim_in,
        dim_out=dim_res1,
        kernel=temp_kernel[0] + [3, 3],
        stride=[1, 2, 2],
        padding=[temp_kernel[0][0] // 2, 1, 1],
        norm_module=norm_module,
        stem_func_name="x3d_stem",
        temporal_window_size=temporal_window_size,
        temporal_fill=temporal_fill,
    )
    modules.append(("s1", s1))

    # blob_in = s1
    dim_in = dim_res1
    dim_out = dim_in
    for stage, block in enumerate(block_basis):
        dim_out = _round_width(block[1], w_mul)
        dim_inner = int(x3d_bottleneck_factor * dim_out)

        n_rep = _round_repeats(block[0], d_mul)
        prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

        s = CoResStage(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_inner=dim_inner,
            temp_kernel_sizes=temp_kernel[1],
            stride=block[2],
            num_blocks=n_rep,
            num_groups=dim_inner if x3d_use_channelwise_3x3x3 else num_groups,
            num_block_temp_kernel=n_rep,
            trans_func=CoX3DTransform,
            stride_1x1=False,
            norm_module=norm_module,
            dilation=1,
            drop_connect_rate=0.0,
            temporal_window_size=temporal_window_size,
            temporal_fill=temporal_fill,
            se_scope=se_scope,
        )
        dim_in = dim_out
        modules.append((prefix, s))

    spat_sz = int(math.ceil(image_size / 32.0))
    head = CoX3DHead(
        dim_in=dim_out,
        dim_inner=dim_inner,
        dim_out=x3d_conv5_dim,
        num_classes=num_classes,
        pool_size=(temporal_window_size, spat_sz, spat_sz),
        dropout_rate=x3d_dropout_rate,
        act_func=x3d_head_activation,
        bn_lin5_on=bool(x3d_head_batchnorm),
        temporal_window_size=temporal_window_size,
        temporal_fill=temporal_fill,
        no_pool=headless,
    )
    modules.append(("head", head))
    seq = co.Sequential(OrderedDict(modules))
    init_weights(seq, x3d_fc_std_init, bool(x3d_final_batchnorm_zero_init))
    return seq


def _round_width(width, multiplier, min_depth=8, divisor=8):
    """Round width of filters based on width multiplier."""
    if not multiplier:
        return width

    width *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * width:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats, multiplier):
    """Round number of layers based on depth multiplier."""
    multiplier = multiplier
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
