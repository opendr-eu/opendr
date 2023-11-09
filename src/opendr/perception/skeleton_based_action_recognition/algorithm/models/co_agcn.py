"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
from collections import OrderedDict
import continual as co
import torch
from torch import nn
from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_base import (
    CoModelBase,
    CoSpatioTemporalBlock,
    init_weights,
)
import numpy as np


class AdaptiveGraphConvolutionMod(nn.Module):
    def __init__(self, in_channels, out_channels, A, bn_momentum=0.1, coff_embedding=4):
        super(AdaptiveGraphConvolutionMod, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        self.a_conv = nn.ModuleList()
        self.b_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            self.a_conv.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.b_conv.append(nn.Conv2d(in_channels, self.inter_c, 1))
            init_weights(self.g_conv[i], bs=self.num_subset)
            init_weights(self.a_conv[i])
            init_weights(self.b_conv[i])

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.graph_attn
        hidden = None
        for i in range(self.num_subset):
            A1 = self.a_conv[i](x).permute(0, 3, 1, 2).contiguous()
            A2 = self.b_conv[i](x)
            # Modified attention within timestep
            A1 = self.soft(torch.einsum("nvct,nctw->nvwt", A1, A2) / self.inter_c)  # N V V T
            A1 = A1 + A[i].unsqueeze(0).unsqueeze(-1)
            z = self.g_conv[i](torch.einsum("nctv,nvwt->nctv", x, A1))
            hidden = z + hidden if hidden is not None else z
        hidden = self.bn(hidden)
        hidden += self.gcn_residual(x)
        return self.relu(hidden)


def CoAdaptiveGraphConvolution(in_channels, out_channels, A, bn_momentum=0.1):
    return co.forward_stepping(AdaptiveGraphConvolutionMod(in_channels, out_channels, A, bn_momentum))


class CoAGcnMod(CoModelBase):
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
        (C_in, T, _, _) = self.input_shape
        A = self.graph.A

        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        self.layers = co.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        CoSpatioTemporalBlock(
                            C_in,
                            64,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=T,
                            residual=False,
                        ),
                    ),
                    (
                        "layer2",
                        CoSpatioTemporalBlock(
                            64,
                            64,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=T - 1 * 8,
                        ),
                    ),
                    (
                        "layer3",
                        CoSpatioTemporalBlock(
                            64,
                            64,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=T - 2 * 8,
                        ),
                    ),
                    (
                        "layer4",
                        CoSpatioTemporalBlock(
                            64,
                            64,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
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
                            CoGraphConv=CoAdaptiveGraphConvolution,
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
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=(T - 4 * 8) - 1 * 8,
                        ),
                    ),
                    (
                        "layer7",
                        CoSpatioTemporalBlock(
                            128,
                            128,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=(T - 4 * 8) - 2 * 8,
                        ),
                    ),
                    (
                        "layer8",
                        CoSpatioTemporalBlock(
                            128,
                            256,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=(T - 4 * 8) - 3 * 8,
                            stride=1,
                        ),
                    ),
                    (
                        "layer9",
                        CoSpatioTemporalBlock(
                            256,
                            256,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=((T - 4 * 8) - 3 * 8) - 1 * 8,
                        ),
                    ),
                    (
                        "layer10",
                        CoSpatioTemporalBlock(
                            256,
                            256,
                            A,
                            CoGraphConv=CoAdaptiveGraphConvolution,
                            padding=0,
                            window_size=((T - 4 * 8) - 3 * 8) - 2 * 8,
                        ),
                    ),
                ]
            )
        )

        # Other layers defined in CoModelBase.on_init_end
        CoModelBase.on_init_end(self)
