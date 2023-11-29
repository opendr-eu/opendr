from __future__ import absolute_import, division, print_function

import torch.nn as nn

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import (
    ConvModule,
    DepthwiseConvModule)


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
        norm_cfg=dict(type="BN"),
        use_depthwise=False,
    ):
        super(Vgg, self).__init__()
        self.num_layers = len(stages_outplanes)
        for layers_args in [stages_outplanes, stages_kernels, stages_strides, stages_padding,
                            maxpool_kernels, maxpool_strides]:
            if len(layers_args) != self.num_layers:
                raise KeyError("Not all convolution args have the same length")
        assert set(out_stages).issubset(range(len(stages_outplanes)))

        conv = DepthwiseConvModule if use_depthwise else ConvModule

        self.out_stages = out_stages

        self.backbone = nn.ModuleList()
        for idx, (ouch, k, s, p, mpk, mps) in enumerate(zip(stages_outplanes, stages_kernels, stages_strides, stages_padding,
                                                        maxpool_kernels, maxpool_strides)):
            inch = 3 if idx == 0 else stages_outplanes[idx - 1]
            pool = nn.MaxPool2d(kernel_size=mpk, stride=mps, padding=mpk // 2) if mpk != 0 else None
            self.backbone.append(conv(inch, ouch, kernel_size=k, stride=s, padding=p, norm_cfg=norm_cfg,
                                      activation=activation, pool=pool))

        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, x):
        y = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self.out_stages:
                y.append(x)
        return y
