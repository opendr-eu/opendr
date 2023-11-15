from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import (
    Conv,
    ConvPool,
    DWConv,
    DWConvPool,
    MultiOutput,
    fuse_modules)


class Vgg(nn.Module):

    def __init__(
        self,
        out_stages=(0, 1, 2, 3),
        stages_outplanes=(8, 8, 6, 6),
        stages_strides=(2, 1, 1, 1),
        stages_kernels=(3, 3, 3, 3),
        stages_padding=(1, 1, 1, 1),
        maxpool_kernels=(0, 0, 0, 0),
        maxpool_strides=(0, 0, 0, 0),
        activation="ReLU",
        use_depthwise=False,
    ):
        super(Vgg, self).__init__()
        self.num_layers = len(stages_outplanes)
        for layers_args in [stages_outplanes, stages_kernels, stages_strides, stages_padding,
                            maxpool_kernels, maxpool_strides]:
            if len(layers_args) != self.num_layers:
                raise KeyError(
                    f"Not all convolution args have the same length")
        assert set(out_stages).issubset(range(len(stages_outplanes)))

        act = act_layers(activation)


        Convs = (DWConv, DWConvPool) if use_depthwise else (Conv, ConvPool)

        self.out_stages = out_stages

        self.backbone = nn.ModuleList()
        for idx, (ouch, k, s, p, mpk, mps) in enumerate(zip(stages_outplanes, stages_kernels, stages_strides, stages_padding,
                                                      maxpool_kernels, maxpool_strides)):
            inch = 3 if idx == 0 else stages_outplanes[idx - 1]
            conv = Convs[1] if mpk != 0 else Convs[0]
            maxpool = nn.MaxPool2d(kernel_size=mpk, stride=mps, padding=mpk // 2)
            self.backbone.append(conv(inch, ouch, k=k, s=s, p=p, act=act, pool=maxpool))
            self.backbone[-1].i = idx
            self.backbone[-1].f = -1

        self.backbone.append(MultiOutput())
        self.backbone[-1].i = -1
        self.backbone[-1].f = self.out_stages

        self.backbone = nn.Sequential(*self.backbone)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        y = []
        for layer in self.backbone:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x if layer.i in self.out_stages else None)
        return x