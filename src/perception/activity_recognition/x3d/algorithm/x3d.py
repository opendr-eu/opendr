""" Adapted from: https://github.com/facebookresearch/SlowFast
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .head_helper import X3DHead
from .resnet_helper import ResStage
from .stem_helper import VideoModelStem
import pytorch_lightning as pl


class X3D(pl.LightningModule):
    """
    X3D model,
    adapted from https://github.com/facebookresearch/SlowFast

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(
        self,
        dim_in: int,
        image_size: int,
        frames_per_clip: int,
        num_classes: int,
        conv1_dim: int,
        conv5_dim: int,
        num_groups: int,
        width_per_group: int,
        width_factor: float,
        depth_factor: float,
        bottleneck_factor: float,
        use_channelwise_3x3x3: bool,
        dropout_rate: float,
        head_activation: str,
        head_batchnorm: bool,
        fc_std_init: float,
        final_batchnorm_zero_init: bool,
        loss_name="cross_entropy",
    ):
        super().__init__()
        self.norm_module = torch.nn.BatchNorm3d
        self.loss_name = loss_name

        exp_stage = 2.0
        self.dim_conv1 = conv1_dim

        self.dim_res2 = (
            _round_width(self.dim_conv1, exp_stage, divisor=8)
            if False  # hparams.X3D.SCALE_RES2
            else self.dim_conv1
        )
        self.dim_res3 = _round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = _round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = _round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]

        num_groups = num_groups
        width_per_group = width_per_group
        dim_inner = num_groups * width_per_group

        w_mul = width_factor
        d_mul = depth_factor
        dim_res1 = _round_width(self.dim_conv1, w_mul)

        # Basis of temporal kernel sizes for each of the stage.
        temp_kernel = [
            [[5]],  # conv1 temporal kernels.
            [[3]],  # res2 temporal kernels.
            [[3]],  # res3 temporal kernels.
            [[3]],  # res4 temporal kernels.
            [[3]],  # res5 temporal kernels.
        ]

        self.s1 = VideoModelStem(
            dim_in=[dim_in],
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        dim_out = dim_in
        for stage, block in enumerate(self.block_basis):
            dim_out = _round_width(block[1], w_mul)
            dim_inner = int(bottleneck_factor * dim_out)

            n_rep = _round_repeats(block[0], d_mul)
            prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

            s = ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner] if use_channelwise_3x3x3 else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
                instantiation="dot_product",
                trans_func_name="x3d_transform",
                stride_1x1=False,
                norm_module=self.norm_module,
                dilation=[1],
                drop_connect_rate=0.0,
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        spat_sz = int(math.ceil(image_size / 32.0))
        self.head = X3DHead(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=conv5_dim,
            num_classes=num_classes,
            pool_size=(frames_per_clip, spat_sz, spat_sz),
            dropout_rate=dropout_rate,
            act_func=head_activation,
            bn_lin5_on=bool(head_batchnorm),
        )
        init_weights(self, fc_std_init, bool(final_batchnorm_zero_init))

    def forward(self, x: Tensor):
        # The original slowfast code was set up to use multiple paths, wrap the input
        x = [x]  # type:ignore
        for module in self.children():
            x = module(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('train/loss', loss)
        self.log('train/acc', _accuracy(x, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('val/loss', loss)
        self.log('val/acc', _accuracy(x, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('test/loss', loss)
        self.log('test/acc', _accuracy(x, y))
        return loss


def _accuracy(x: Tensor, y: Tensor):
    return torch.sum(x.argmax(dim=1) == y) / len(y)


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


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
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
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
