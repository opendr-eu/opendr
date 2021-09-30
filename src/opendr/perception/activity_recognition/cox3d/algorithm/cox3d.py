"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolit with permission from the author.
"""

import math
import torch

from .conv import ConvCo3d
from .delay import Delay
from .pooling import AdaptiveAvgPoolCo3d, AvgPoolCo3d
from .se import ReSe
from .utils import FillMode, unsqueezed, warn_once_if
from opendr.perception.activity_recognition.x3d.algorithm.operators import Swish
from opendr.perception.activity_recognition.x3d.algorithm.x3d import (
    _accuracy, _round_width, _round_repeats, init_weights
)
import torch.nn.functional as F
import pytorch_lightning as pl


class CoX3DTransform(torch.nn.Module):
    """
    Recursive X3D transformation: 1x1x1, Tx3x3 (channelwise, num_groups=dim_in), 1x1x1,
        augmented with (optional) SE (squeeze-excitation) on the 3x3x3 output.
        T is the temporal kernel size (defaulting to 3)
    """

    def __init__(
        self,
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
        temporal_fill: FillMode = "replicate",
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
        super(CoX3DTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._se_ratio = se_ratio
        self._swish_inner = swish_inner
        self._stride_1x1 = stride_1x1
        self._block_idx = block_idx
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
            temporal_window_size,
            temporal_fill,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
        temporal_window_size,
        temporal_fill="replicate",
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1x1, BN, ReLU.
        self.a = unsqueezed(
            torch.nn.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, 1, 1),
                stride=(1, str1x1, str1x1),
                padding=(0, 0, 0),
                bias=False,
            )
        )
        self.a_bn = unsqueezed(
            norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        )

        self.a_relu = torch.nn.ReLU(inplace=self._inplace_relu)

        # Tx3x3, BN, ReLU.
        self.b = ConvCo3d(
            dim_inner,
            dim_inner,
            kernel_size=(self.temp_kernel_size, 3, 3),
            stride=(1, str3x3, str3x3),
            padding=(int(self.temp_kernel_size // 2), dilation, dilation),
            groups=num_groups,
            bias=False,
            dilation=(1, dilation, dilation),
            temporal_fill=temporal_fill,
        )

        self.b_bn = unsqueezed(
            norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        )

        # Apply SE attention or not
        use_se = True if (self._block_idx + 1) % 2 else False
        if self._se_ratio > 0.0 and use_se:
            self.se = ReSe(
                temporal_window_size,
                dim_in=dim_inner,
                ratio=self._se_ratio,
                temporal_fill=temporal_fill,
            )

        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = torch.nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = unsqueezed(
            torch.nn.Conv3d(
                dim_inner,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            )
        )
        self.c_bn = unsqueezed(
            norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    def forward3d(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.children():
            if hasattr(block, "forward3d"):
                x = block.forward3d(x)
            else:
                x = block.forward(x)
        return x


class ReResBlock(torch.nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
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
        temporal_fill: FillMode = "replicate",
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
        super(ReResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = unsqueezed(
                torch.nn.Conv3d(
                    dim_in,
                    dim_out,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    padding=0,
                    bias=False,
                    dilation=1,
                )
            )
            self.branch1_bn = unsqueezed(
                norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
            )
        self.branch2 = trans_func(
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
        )
        self.relu = torch.nn.ReLU(self._inplace_relu)
        # temporal_fill="replicate" works much better than "zeros" here
        self.delay = Delay(temp_kernel_size - 1, temporal_fill)

    def _drop_connect(self, x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        delayed_x = self.delay(x)
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            output = self.branch1_bn(self.branch1(delayed_x)) + f_x
        else:
            output = delayed_x + f_x
        output = self.relu(output)
        return output

    def forward3d(self, x):
        f_x = self.branch2.forward3d(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn.forward3d(self.branch1.forward3d(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class ReResStage(torch.nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        trans_func_name="x3d_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=torch.nn.BatchNorm3d,
        drop_connect_rate=0.0,
        temporal_window_size: int = 4,
        temporal_fill: FillMode = "replicate",
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ReResStage, self).__init__()
        assert trans_func_name == "x3d_transform"
        assert nonlocal_inds == [[]], "Nonlocal network not supported currently."
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]] +
            [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            ) == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            CoX3DTransform,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
            temporal_window_size,
            temporal_fill,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
        temporal_window_size,
        temporal_fill,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Construct the block.
                res_block = ReResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                    temporal_window_size=temporal_window_size,
                    temporal_fill=temporal_fill,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output

    def forward3d(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m.forward3d(x)
            output.append(x)

        return output


class CoX3DHead(torch.nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=torch.nn.BatchNorm3d,
        bn_lin5_on=False,
        temporal_window_size: int = 4,
        temporal_fill: FillMode = "replicate",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
                default is torch.nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(CoX3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(
            dim_in,
            dim_inner,
            dim_out,
            norm_module,
            temporal_window_size,
            temporal_fill,
        )

    def _construct_head(
        self,
        dim_in,
        dim_inner,
        dim_out,
        norm_module,
        temporal_window_size,
        temporal_fill,
    ):

        self.conv_5 = unsqueezed(
            torch.nn.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            )
        )
        self.conv_5_bn = unsqueezed(
            norm_module(num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt)
        )
        self.conv_5_relu = torch.nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = AdaptiveAvgPoolCo3d(1, temporal_fill, (1, 1))
        else:
            self.avg_pool = AvgPoolCo3d(
                self.pool_size[0], temporal_fill, 1, self.pool_size[1:], stride=1
            )

        self.lin_5 = unsqueezed(
            torch.nn.Conv3d(
                dim_inner,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            )
        )
        if self.bn_lin5_on:
            self.lin_5_bn = unsqueezed(
                norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
            )
        self.lin_5_relu = torch.nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = torch.nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = torch.nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = unsqueezed(torch.nn.Softmax(dim=4))
        elif self.act_func == "sigmoid":
            self.act = torch.nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        # x = x.permute((0, 2, 3, 4, 1))
        # Compatible with recursive conversion:
        x = x.unsqueeze(-1).transpose(-1, 1).squeeze(1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2])

        x = x.view(x.shape[0], -1)
        return x

    def forward3d(self, inputs):
        # In its current design the X3D head is only usable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5.forward3d(inputs[0])
        x = self.conv_5_bn.forward3d(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool.forward3d(x)

        x = self.lin_5.forward3d(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn.forward3d(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        # x = x.permute((0, 2, 3, 4, 1))
        # Compatible with recursive conversion:
        x = x.unsqueeze(-1).transpose(-1, 1).squeeze(1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            x = self.act.forward3d(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class CoX3DStem(torch.nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=torch.nn.BatchNorm3d,
        temporal_window_size: int = 4,
        temporal_fill: FillMode = "replicate",
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

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
        super(CoX3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module, temporal_fill)

    def _construct_stem(self, dim_in, dim_out, norm_module, temporal_fill):
        self.conv_xy = unsqueezed(
            torch.nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, self.kernel[1], self.kernel[2]),
                stride=(1, self.stride[1], self.stride[2]),
                padding=(0, self.padding[1], self.padding[2]),
                bias=False,
            )
        )
        self.conv = ConvCo3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
            temporal_fill=temporal_fill,
        )
        self.bn = unsqueezed(
            norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        )
        self.relu = torch.nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def forward3d(self, x):
        x = self.conv_xy.forward3d(x)
        x = self.conv.forward3d(x)
        x = self.bn.forward3d(x)
        x = self.relu(x)
        return x


class ReVideoModelStem(torch.nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=torch.nn.BatchNorm3d,
        stem_func_name="x3d_stem",
        temporal_window_size: int = 4,
        temporal_fill: FillMode = "replicate",
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
                default is torch.nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(ReVideoModelStem, self).__init__()

        assert (
            len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)}) == 1
        ), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        assert (
            stem_func_name == "x3d_stem"
        ), "Currently, only 'x3d_stem' stem func is implemented."
        for pathway in range(len(dim_in)):
            stem = CoX3DStem(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
                temporal_window_size=temporal_window_size,
                temporal_fill=temporal_fill,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x

    def forward3d(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m.forward3d(x[pathway])
        return x


class CoX3D(pl.LightningModule):
    """
    Recurrent X3D model,
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
        temporal_fill: FillMode = "replicate",
        loss_name="cross_entropy",
    ):
        super().__init__()
        self.temporal_window_size = frames_per_clip
        self.loss_name = loss_name

        exp_stage = 2.0
        self.norm_module = torch.nn.BatchNorm3d  # Will be unsqueezed
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

        self.s1 = ReVideoModelStem(
            dim_in=[dim_in],
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            # padding=[[0, 1, 1]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
            temporal_window_size=frames_per_clip,
            temporal_fill=temporal_fill,
        )

        dim_in = dim_res1
        dim_out = dim_in
        for stage, block in enumerate(self.block_basis):
            dim_out = _round_width(block[1], w_mul)
            dim_inner = int(bottleneck_factor * dim_out)

            n_rep = _round_repeats(block[0], d_mul)
            prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

            s = ReResStage(
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
                temporal_window_size=frames_per_clip,
                temporal_fill=temporal_fill,
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        spat_sz = int(math.ceil(image_size / 32.0))
        self.head = CoX3DHead(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=conv5_dim,
            num_classes=num_classes,
            pool_size=(frames_per_clip, spat_sz, spat_sz),
            dropout_rate=dropout_rate,
            act_func=head_activation,
            bn_lin5_on=bool(head_batchnorm),
            temporal_window_size=4,
            temporal_fill=temporal_fill,
        )
        init_weights(self, fc_std_init, bool(final_batchnorm_zero_init))

    def receptive_field(self):
        if not hasattr(self, "_receptive_field"):
            self._receptive_field = sum(
                [m.kernel_size[0] - 1 for m in self.modules() if "ConvCo3d" in str(type(m))]
            ) + self.temporal_window_size
        return self._receptive_field

    def clean_model_state(self):
        for m in self.modules():
            if hasattr(m, "clean_state"):
                m.clean_state()

    def forward(self, x: torch.Tensor):
        """Pass Image through the network

        Args:
            x (torch.Tensor): Image batch of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Class prediction
        """
        if getattr(self, "_current_input_shape", None) != x.shape:
            self.clean_model_state()
            self._current_input_shape = x.shape

        x = [x]
        for module in self.children():
            x = module(x)
        return x

    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """Pass Video through the network. Only the prediction for the last frame is returned.

        Args:
            x (torch.Tensor): Video batch of shape (B, 3, T, H, W)

        Returns:
            torch.Tensor: Class prediction
        """
        T = x.shape[2]

        warn_once_if(
            cond=self.receptive_field() > T,
            msg=(
                "Input clip has length smaller than the model receptive field. "
                "This may yield worse predictions than expected because the network "
                "is operating within its transient response."
            )
        )

        for i in range(T - 1):
            self.forward(x[:, :, i])

        result = self.forward(x[:, :, -1])
        return result

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward_video(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('train/loss', loss)
        self.log('train/acc', _accuracy(x, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward_video(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('val/loss', loss)
        self.log('val/acc', _accuracy(x, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward_video(x)
        loss = getattr(F, self.loss_name, F.cross_entropy)(x, y)
        self.log('test/loss', loss)
        self.log('test/acc', _accuracy(x, y))
        return loss
