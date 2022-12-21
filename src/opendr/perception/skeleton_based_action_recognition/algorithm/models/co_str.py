"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
from collections import OrderedDict
import continual as co
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_base import (
    CoModelBase,
    CoSpatioTemporalBlock,
)


class CoSTrMod(CoModelBase):
    def __init__(
        self,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph_type="ntu",
        sequence_len: int = 300,
        num_classes: int = 60,
        loss_name="cross_entropy",
    ):
        CoModelBase.__init__(
            self, num_point, num_person, in_channels, graph_type, sequence_len, num_classes, loss_name
        )

        # Shapes: num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, V, S) = self.input_shape
        A = self.graph.A

        def CoGcnUnitAttention(in_channels, out_channels, A, bn_momentum=0.1):
            return co.forward_stepping(
                GcnUnitAttention(in_channels, out_channels, A, bn_momentum, num_point=V)
            )

        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        self.layers = co.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        CoSpatioTemporalBlock(C_in, 64, A, padding=0, window_size=T, residual=False),
                    ),
                    ("layer2", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 1 * 8)),
                    ("layer3", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 2 * 8)),
                    (
                        "layer4",
                        CoSpatioTemporalBlock(
                            64,
                            64,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=T - 3 * 8,
                        ),
                    ),
                    (
                        "layer5",
                        CoSpatioTemporalBlock(
                            64,
                            128,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=T - 4 * 8,
                            stride=1,
                        ),
                    ),
                    (
                        "layer6",
                        CoSpatioTemporalBlock(
                            128,
                            128,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=(T - 4 * 8) / 2 - 1 * 8,
                        ),
                    ),
                    (
                        "layer7",
                        CoSpatioTemporalBlock(
                            128,
                            128,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=(T - 4 * 8) / 2 - 2 * 8,
                        ),
                    ),
                    (
                        "layer8",
                        CoSpatioTemporalBlock(
                            128,
                            256,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=(T - 4 * 8) / 2 - 3 * 8,
                            stride=1,
                        ),
                    ),
                    (
                        "layer9",
                        CoSpatioTemporalBlock(
                            256,
                            256,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8,
                        ),
                    ),
                    (
                        "layer10",
                        CoSpatioTemporalBlock(
                            256,
                            256,
                            A,
                            CoGraphConv=CoGcnUnitAttention,
                            padding=0,
                            window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8,
                        ),
                    ),
                ]
            )
        )

        # Other layers defined in CoModelBase.on_init_end
        CoModelBase.on_init_end(self)


class SpatialAttention(nn.Module):
    """
    This class implements Spatial Attention.
    Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
    """

    def __init__(
        self,
        in_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        complete,
        relative,
        layer,
        A,
        more_channels,
        drop_connect,
        adjacency,
        num,
        num_point,
        shape=25,
        stride=1,
        last_graph=False,
        data_normalization=True,
        skip_conn=True,
        visualization=True,
    ):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.complete = complete
        self.kernel_size = 1
        self.dk = dk
        self.dv = dv
        self.num = num
        self.layer = layer
        self.more_channels = more_channels
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.adjacency = adjacency
        self.Nh = Nh
        self.num_point = num_point
        self.A = A[0] + A[1] + A[2]
        if self.adjacency:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        self.shape = shape
        self.relative = relative
        self.last_graph = last_graph
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        if self.more_channels:

            self.qkv_conv = nn.Conv2d(
                self.in_channels,
                (2 * self.dk + self.dv) * self.Nh // self.num,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
            )
        else:
            self.qkv_conv = nn.Conv2d(
                self.in_channels,
                2 * self.dk + self.dv,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
            )
        if self.more_channels:

            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            # Two parameters are initialized in order to implement relative positional encoding
            # One weight repeated over the diagonal
            # V^2-V+1 paramters in positions outside the diagonal
            if self.more_channels:
                self.key_rel = nn.Parameter(
                    torch.randn(
                        ((self.num_point ** 2) - self.num_point, self.dk // self.num),
                        requires_grad=True,
                    )
                )
            else:
                self.key_rel = nn.Parameter(
                    torch.randn(
                        ((self.num_point ** 2) - self.num_point, self.dk // Nh),
                        requires_grad=True,
                    )
                )
            if self.more_channels:
                self.key_rel_diagonal = nn.Parameter(
                    torch.randn((1, self.dk // self.num), requires_grad=True)
                )
            else:
                self.key_rel_diagonal = nn.Parameter(
                    torch.randn((1, self.dk // self.Nh), requires_grad=True)
                )

    def forward(self, x):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        # In this version, the adjacency matrix is weighted and added to the attention logits of transformer to add
        # information of the original skeleton structure
        if self.adjacency:
            logits = logits.reshape(-1, V, V)
            M, V, V = logits.shape
            A = self.A
            A *= self.mask
            A = A.unsqueeze(0).expand(M, V, V)
            logits = logits + A
            logits = logits.reshape(B, self.Nh, V, V)

        # Relative positional encoding is used or not
        if self.relative:
            rel_logits = self.relative_logits(q)
            logits_sum = torch.add(logits, rel_logits)

        # Calculate weights
        if self.relative:
            weights = F.softmax(logits_sum, dim=-1)
        else:
            weights = F.softmax(logits, dim=-1)

        # Drop connect implementation to avoid overfitting
        if self.drop_connect and self.training:
            mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * V, device=x.device))
            mask = mask.reshape(B, self.Nh, V).unsqueeze(2).expand(B, self.Nh, V, V)
            weights = weights * mask
            weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        if not self.more_channels:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
        else:
            attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.num))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        # if self.more_channels=True, to each head is assigned dk*self.Nh//self.num channels
        if self.more_channels:
            q, k, v = torch.split(
                qkv,
                [
                    dk * self.Nh // self.num,
                    dk * self.Nh // self.num,
                    dv * self.Nh // self.num,
                ],
                dim=1,
            )
        else:
            q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        if self.more_channels:
            flat_q = torch.reshape(q, (N, Nh, dk // self.num, T * V))
            flat_k = torch.reshape(k, (N, Nh, dk // self.num, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.num, T * V))
        else:
            flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
            flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
            flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, T, V = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
        q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))

        # q used to multiply for the embedding of the parameter on the diagonal
        q = torch.reshape(q, (B * Nh * T, V, dk))
        # key_rel_diagonal: (1, dk) -> (V, dk)
        param_diagonal = self.key_rel_diagonal.expand((V, dk))
        rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
        return rel_logits

    def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
        # compute relative logits along one dimension
        # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)

        # (B*Nh*1, V^2-V)
        rel_logits = torch.einsum("bmd,md->bm", q_first, rel_k)
        # (B*Nh*1, V)
        rel_logits_diagonal = torch.einsum("bmd,md->bm", q, param_diagonal)

        # reshapes to obtain Srel
        rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
        return rel_logits

    def rel_to_abs(self, rel_logits, rel_logits_diagonal):
        B, L = rel_logits.size()
        B, V = rel_logits_diagonal.size()

        # (B, V-1, V) -> (B, V, V)
        rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
        row_pad = torch.zeros(B, 1, V).to(rel_logits)
        rel_logits = torch.cat((rel_logits, row_pad), dim=1)

        # concat the other embedding on the left
        # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
        rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
        rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
        rel_logits = torch.reshape(rel_logits, (B, V + 1, V))

        # slice
        flat_sliced = rel_logits[:, :V, :]
        final_x = torch.reshape(flat_sliced, (B, V, V))
        return final_x


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n = n * k
    module.weight.data.normal_(0, math.sqrt(2.0 / n))


class GcnUnitAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        num=4,
        dv_factor=0.25,
        dk_factor=0.25,
        Nh=8,
        complete=True,
        relative=False,
        only_attention=True,
        layer=0,
        more_channels=False,
        drop_connect=True,
        data_normalization=True,
        skip_conn=True,
        adjacency=False,
        num_point=25,
        padding=0,
        kernel_size=1,
        stride=1,
        bn_flag=True,
        t_dilation=1,
        last_graph=False,
        visualization=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.visualization = visualization
        self.in_channels = in_channels
        self.more_channels = more_channels
        self.drop_connect = drop_connect
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        # print("Nh ", Nh)
        # print("Dv ", dv_factor)
        # print("Dk ", dk_factor)

        self.last_graph = last_graph
        if not only_attention:
            self.out_channels = out_channels - int((out_channels) * dv_factor)
        else:
            self.out_channels = out_channels
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        self.bn = nn.BatchNorm2d(out_channels)
        self.only_attention = only_attention
        self.bn_flag = bn_flag
        self.layer = layer

        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        # Each Conv2d unit implements 2d convolution to weight every single partition (filter size 1x1)
        # There is a convolutional unit for each partition
        # This is done only in the case in which Spatial Transformer and Graph Convolution are concatenated

        if not self.only_attention:
            self.g_convolutions = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels,
                        self.out_channels,
                        kernel_size=(kernel_size, 1),
                        padding=(padding, 0),
                        stride=(stride, 1),
                        dilation=(t_dilation, 1),
                    )
                    for i in range(self.A.size()[0])
                ]
            )
            for conv in self.g_convolutions:
                conv_init(conv)

            self.attention_conv = SpatialAttention(
                in_channels=self.in_channels,
                kernel_size=1,
                dk=int(out_channels * dk_factor),
                dv=int(out_channels * dv_factor),
                Nh=Nh,
                complete=complete,
                relative=relative,
                stride=stride,
                layer=self.layer,
                A=self.A,
                num=num,
                more_channels=self.more_channels,
                drop_connect=self.drop_connect,
                data_normalization=self.data_normalization,
                skip_conn=self.skip_conn,
                adjacency=self.adjacency,
                visualization=self.visualization,
                num_point=self.num_point,
            )
        else:
            self.attention_conv = SpatialAttention(
                in_channels=self.in_channels,
                kernel_size=1,
                dk=int(out_channels * dk_factor),
                dv=int(out_channels),
                Nh=Nh,
                complete=complete,
                relative=relative,
                stride=stride,
                last_graph=self.last_graph,
                layer=self.layer,
                A=self.A,
                num=num,
                more_channels=self.more_channels,
                drop_connect=self.drop_connect,
                data_normalization=self.data_normalization,
                skip_conn=self.skip_conn,
                adjacency=self.adjacency,
                visualization=self.visualization,
                num_point=self.num_point,
            )

    def forward(self, x):
        # N: number of samples, equal to the batch size
        # C: number of channels, in our case 3 (coordinates x, y, z)
        # T: number of frames
        # V: number of nodes
        N, C, T, V = x.size()
        x_sum = x
        if self.data_normalization:
            x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
            x = self.data_bn(x)
            x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        # Learnable parameter
        A = self.A

        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)

        # Spatial Transformer
        attn_out = self.attention_conv(xa)
        # N, T, C, V > N, C, T, V
        attn_out = attn_out.reshape(N, T, -1, V).permute(0, 2, 1, 3)

        if not self.only_attention:

            # For each partition multiplies for the input and applies convolution 1x1 to the result to weight each partition
            for i, partition in enumerate(A):
                # print(partition)
                # NCTxV
                xp = x.reshape(-1, V)
                # (NCTxV)*(VxV)
                xp = xp.mm(partition.float())
                # NxCxTxV
                xp = xp.reshape(N, C, T, V)

                if i == 0:
                    y = self.g_convolutions[i](xp)
                else:
                    y = y + self.g_convolutions[i](xp)

            # Concatenate on the channel dimension the two convolutions
            y = torch.cat((y, attn_out), dim=1)
        else:
            if self.skip_conn and self.in_channels == self.out_channels:
                y = attn_out + x_sum
            else:
                y = attn_out
        if self.bn_flag:
            y = self.bn(y)

        y = self.relu(y)

        return y
