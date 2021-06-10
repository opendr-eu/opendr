import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from math import ceil

from mmdet.core import auto_fp16, force_fp32
from mmdet.ops import ConvModule, DepthwiseSeparableConvModule, build_upsample_layer
from ..registry import HEADS

class MC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, size=(x.shape[-2]*2, x.shape[-1]*2), 
                          mode='bilinear', align_corners=False)

        return x


class LSFE(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class DPC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        dilations = [(1,6), (1,1), (6,21), (18,15), (6,3)]

        for i in range(5):
            padding = dilations[i]
            self.convs.append(
                DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.in_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dilation=dilations[i]))
        self.conv = ConvModule(
                    self.in_channels*5,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)

    def forward(self, x):
        x = self.convs[0](x)
        x1 = self.convs[1](x)
        x2 = self.convs[2](x)
        x3 = self.convs[3](x)
        x4 = self.convs[4](x3)
        x = torch.cat([
            x,
            x1,
            x2,
            x3,
            x4 
            ], dim=1)

        x = self.conv(x) 
        return x



@HEADS.register_module
class EfficientPSSemanticHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 conv_out_channels=128,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=1.0,
                 ohem = 0.25,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(EfficientPSSemanticHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.ohem = ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
    
        if self.ohem is not None:
            assert (self.ohem >= 0 and self.ohem < 1)

        self.lateral_convs_ss = nn.ModuleList()
        self.lateral_convs_ls = nn.ModuleList()
        self.aligning_convs = nn.ModuleList()
        self.ss_idx = [3,2]
        self.ls_idx = [1,0] 
        for i in range(2):
            self.lateral_convs_ss.append(
                DPC(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(2):
            self.lateral_convs_ls.append(
                LSFE(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(2):
            self.aligning_convs.append(
                  MC(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.conv_logits = nn.Conv2d(conv_out_channels * 4, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, feats):
        feats = list(feats)
        ref_size = tuple(feats[0].shape[-2:])
        for idx, lateral_conv_ss in zip(self.ss_idx, self.lateral_convs_ss):
            feats[idx] = lateral_conv_ss(feats[idx])
       
        x = self.aligning_convs[0](feats[self.ss_idx[1]] + F.interpolate(
                    feats[self.ss_idx[0]], size=tuple(feats[self.ss_idx[1]].shape[-2:]), 
                    mode='bilinear', align_corners=False))

        for idx, lateral_conv_ls in zip(self.ls_idx, self.lateral_convs_ls):
            feats[idx] = lateral_conv_ls(feats[idx])
            feats[idx] = feats[idx] + F.interpolate(
                      x, size=tuple(feats[idx].shape[-2:]), 
                      mode='bilinear', align_corners=False)
            if idx != 0:
                x = self.aligning_convs[1](feats[idx])
        
        for i in range(1,4):
            feats[i] = F.interpolate(
                      feats[i], size=ref_size, 
                      mode='bilinear', align_corners=False)

        x = torch.cat(feats, dim=1)
        x = self.conv_logits(x)
        x = F.interpolate(
                      x, size=(ref_size[0]*4, ref_size[1]*4), 
                      mode='bilinear', align_corners=False)

        return x


    def loss(self, mask_pred, labels):
        loss = dict()
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg = loss_semantic_seg.view(-1)

        if self.ohem is not None:
            top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
            if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
 
        loss_semantic_seg = loss_semantic_seg.mean()
        loss_semantic_seg *= self.loss_weight
        loss['loss_semantic_seg'] = loss_semantic_seg
        return loss
