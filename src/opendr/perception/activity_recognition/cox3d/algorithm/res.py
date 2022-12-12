from collections import OrderedDict
from typing import Callable

import continual as co
import torch
from continual import PaddingMode
from torch import nn


def CoResBlock(
    dim_in,
    dim_out,
    temp_kernel_size,
    stride,
    trans_func,
    dim_inner,
    num_groups=1,
    stride_1x1=False,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    dilation=1,
    norm_module=torch.nn.BatchNorm3d,
    block_idx=0,
    drop_connect_rate=0.0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",  # "clip" or "frame"
):
    """
    ResBlock class constructs redisual blocks. More details can be found in:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
        "Deep residual learning for image recognition."
        https://arxiv.org/abs/1512.03385
    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temp_kernel_size (int): the temporal kernel sizes of the middle
            convolution in the bottleneck.
        stride (int): the stride of the bottleneck.
        trans_func (string): transform function to be used to construct the
            bottleneck.
        dim_inner (int): the inner dimension of the block.
        num_groups (int): number of groups for the convolution. num_groups=1
            is for standard ResNet like networks, and num_groups>1 is for
            ResNeXt like networks.
        stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
            apply stride to the 3x3 conv.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dilation (int): size of dilation.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
        drop_connect_rate (float): basic rate at which blocks are dropped,
            linearly increases from input to output blocks.
    """
    branch2 = trans_func(
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=stride_1x1,
        inplace_relu=inplace_relu,
        dilation=dilation,
        norm_module=norm_module,
        block_idx=block_idx,
        temporal_window_size=temporal_window_size,
        temporal_fill=temporal_fill,
        se_scope=se_scope,
    )

    def _is_training(module: nn.Module) -> bool:
        return module.training

    def _drop_connect(x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    if drop_connect_rate > 0:
        drop = [("drop", co.Conditional(_is_training, co.Lambda(_drop_connect)))]
    else:
        drop = []

    main_stream = co.Sequential(
        OrderedDict(
            [
                ("branch2", branch2),
                *drop,
            ]
        )
    )

    if (dim_in == dim_out) and (stride == 1):
        residual_stream = co.Delay(main_stream.delay)
    else:
        residual_stream = co.Sequential(
            OrderedDict(
                [
                    (
                        "branch1",
                        co.Conv3d(
                            dim_in,
                            dim_out,
                            kernel_size=1,
                            stride=(1, stride, stride),
                            padding=0,
                            bias=False,
                            dilation=1,
                        ),
                    ),
                    (
                        "branch1_bn",
                        norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt),
                    ),
                ]
            )
        )

    return co.Sequential(
        co.BroadcastReduce(residual_stream, main_stream, reduce="sum"),
        nn.ReLU(),
    )


def CoResStage(
    dim_in: int,
    dim_out: int,
    stride: int,
    temp_kernel_sizes: int,
    num_blocks: int,
    dim_inner: int,
    num_groups: int,
    num_block_temp_kernel: int,
    dilation: int,
    trans_func: Callable,
    stride_1x1=False,
    inplace_relu=True,
    norm_module=torch.nn.BatchNorm3d,
    drop_connect_rate=0.0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",
    *args,
    **kwargs,
):
    """
    Create a Continual Residual X3D Stage.

    Note: Compared to the original implementation of X3D, we discard the
    obsolete handling of the multiple pathways and the non-local mechanism.

    Args:
        dim_in (int): channel dimensions of the input.
        dim_out (int): channel dimensions of the output.
        temp_kernel_sizes (int): temporal kernel sizes of the
            convolution in the bottleneck.
        stride (int): stride of the bottleneck.
        num_blocks (int): numbers of blocks.
        dim_inner (int): inner channel dimensions of the input.
        num_groups (int): number of groups for the convolution.
            num_groups=1 is for standard ResNet like networks, and
            num_groups>1 is for ResNeXt like networks.
        num_block_temp_kernel (int): extent the temp_kernel_sizes to
            num_block_temp_kernel blocks, then fill temporal kernel size
            of 1 for the rest of the layers.
        dilation (int): size of dilation.
        trans_func (Callable): transformation function to apply on the network.
        norm_module (nn.Module): nn.Module for the normalization layer. The
            default is nn.BatchNorm3d.
        drop_connect_rate (float): basic rate at which blocks are dropped,
            linearly increases from input to output blocks.
    """

    assert num_block_temp_kernel <= num_blocks

    temp_kernel_sizes = (temp_kernel_sizes * num_blocks)[:num_block_temp_kernel] + (
        [1] * (num_blocks - num_block_temp_kernel)
    )

    return co.Sequential(
        OrderedDict(
            [
                (
                    f"pathway0_res{i}",
                    CoResBlock(
                        dim_in=dim_in if i == 0 else dim_out,
                        dim_out=dim_out,
                        temp_kernel_size=temp_kernel_sizes[i],
                        stride=stride if i == 0 else 1,
                        trans_func=trans_func,
                        dim_inner=dim_inner,
                        num_groups=num_groups,
                        stride_1x1=stride_1x1,
                        inplace_relu=inplace_relu,
                        dilation=dilation,
                        norm_module=norm_module,
                        block_idx=i,
                        drop_connect_rate=drop_connect_rate,
                        temporal_window_size=temporal_window_size,
                        temporal_fill=temporal_fill,
                        se_scope=se_scope,
                    ),
                )
                for i in range(num_blocks)
            ]
        )
    )


def c2_msra_fill(module: torch.nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        torch.nn.init.constant_(module.bias, 0)


def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, co.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.BatchNorm2d):
            if (
                hasattr(m, "transform_final_bn") and
                m.transform_final_bn and
                zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
