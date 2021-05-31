import torch
from torch import nn
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.utils import get_paddings_indicator
from opendr.perception.object_detection_3d.voxel_object_detection_3d.\
    second_detector.torchplus_tanet.tools import change_default_args
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.torchplus_tanet.nn import (
    Empty, GroupNorm, Sequential
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.models.pointpillars import PFNLayer
import numpy as np

import yaml
from easydict import EasyDict as edict

cfg = edict(yaml.safe_load('''
Network: TANet      # RefineDet ...


# config for extracting the voxel feature with TA module
TA:

    INPUT_C_DIM: 9
    BOOST_C_DIM: 64  # or 32
    NUM_POINTS_IN_VOXEL: 100
    REDUCTION_R: 8
    # Note: Our released model need set " USE_PACA_WEIGHT: False"
    # When training model, setting " USE_PACA_WEIGHT: True" may be more stable
    USE_PACA_WEIGHT: False #True

PSA:
    C_Bottle: 128
    C_Reudce: 32
'''))


def set_tanet_config(path):
    global cfg

    filename = path
    with open(filename, "r") as f:
        cfg = edict(yaml.load(f))


# Point-wise attention for each voxel
class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa),
        )

    def forward(self, x):
        b, w, _ = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)
        out1 = self.fc(y).view(b, w, 1)
        return out1


# Channel-wise attention for each voxel
class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca),
        )

    def forward(self, x):
        b, _, c = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(b, c)
        y = self.fc(y).view(b, 1, c)
        return y


# Point-wise attention for each voxel
class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer(dim_pa, dim_pa // reduction_r)
        self.ca = CALayer(dim_ca, dim_ca // reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        pa_weight = self.pa(x)
        ca_weight = self.ca(x)
        paca_weight = torch.mul(pa_weight, ca_weight)
        paca_normal_weight = self.sig(paca_weight)
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight


# Voxel-wise attention for each voxel
class VALayer(nn.Module):
    def __init__(self, c_num, p_num):
        super(VALayer, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_num + 3, 1),
                                 nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(p_num, 1),
            nn.ReLU(inplace=True)
        )

        self.sigmod = nn.Sigmoid()

    def forward(self, voxel_center, paca_feat):
        """
        :param voxel_center: size (K,1,3)
        :param SACA_Feat: size (K,N,C)
        :return: voxel_attention_weight: size (K,1,1)
        """
        voxel_center_repeat = voxel_center.repeat(1, paca_feat.shape[1], 1)
        # print(voxel_center_repeat.shape)
        voxel_feat_concat = torch.cat([paca_feat, voxel_center_repeat],
                                      dim=-1)  # K,N,C---> K,N,(C+3)

        feat_2 = self.fc1(voxel_feat_concat)  # K,N,(C+3)--->K,N,1
        feat_2 = feat_2.permute(0, 2, 1).contiguous()  # K,N,1--->K,1,N

        voxel_feat_concat = self.fc2(feat_2)  # K,1,N--->K,1,1

        voxel_attention_weight = self.sigmod(voxel_feat_concat)  # K,1,1

        return voxel_attention_weight


class VoxelFeature_TA(nn.Module):
    def __init__(
        self,
        dim_ca=cfg.TA.INPUT_C_DIM,
        dim_pa=cfg.TA.NUM_POINTS_IN_VOXEL,
        reduction_r=cfg.TA.REDUCTION_R,
        boost_c_dim=cfg.TA.BOOST_C_DIM,
        use_paca_weight=cfg.TA.USE_PACA_WEIGHT,
    ):
        super(VoxelFeature_TA, self).__init__()
        self.PACALayer1 = PACALayer(dim_ca=dim_ca,
                                    dim_pa=dim_pa,
                                    reduction_r=reduction_r)
        self.PACALayer2 = PACALayer(dim_ca=boost_c_dim,
                                    dim_pa=dim_pa,
                                    reduction_r=reduction_r)
        self.voxel_attention1 = VALayer(c_num=dim_ca, p_num=dim_pa)
        self.voxel_attention2 = VALayer(c_num=boost_c_dim, p_num=dim_pa)
        self.use_paca_weight = use_paca_weight
        self.FC1 = nn.Sequential(
            nn.Linear(2 * dim_ca, boost_c_dim),
            nn.ReLU(inplace=True),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(boost_c_dim, boost_c_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_center, x):
        paca1, paca_normal_weight1 = self.PACALayer1(x)
        voxel_attention1 = self.voxel_attention1(voxel_center, paca1)
        if self.use_paca_weight:
            paca1_feat = voxel_attention1 * paca1 * paca_normal_weight1
        else:
            paca1_feat = voxel_attention1 * paca1
        out1 = torch.cat([paca1_feat, x], dim=2)
        out1 = self.FC1(out1)

        paca2, paca_normal_weight2 = self.PACALayer2(out1)
        voxel_attention2 = self.voxel_attention2(voxel_center, paca2)
        if self.use_paca_weight:
            paca2_feat = voxel_attention2 * paca2 * paca_normal_weight2
        else:
            paca2_feat = voxel_attention2 * paca2
        out2 = out1 + paca2_feat
        out = self.FC2(out2)

        return out


# PillarFeature_TANet is modified from pointpillars.PillarFeatureNet
# by introducing Triple Attention
class PillarFeature_TANet(nn.Module):
    def __init__(
            self,
            num_input_features=4,
            use_norm=True,
            num_filters=(64, ),
            with_distance=False,
            voxel_size=(0.2, 0.2, 4),
            pc_range=(0, -40, -3, 70.4, 40, 1),
    ):
        """
        Pillar Feature Net with Tripe attention.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeature_TANet"
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_input_features = cfg.TA.BOOST_C_DIM

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)

        self.VoxelFeature_TA = VoxelFeature_TA()
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters,
                         out_filters,
                         use_norm,
                         last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        features = self.VoxelFeature_TA(points_mean, features)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


# Our Coarse-to-Fine network
class PSA(nn.Module):
    def __init__(
        self,
        use_norm=True,
        num_class=2,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        num_filters=[128, 128, 256],
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[256, 256, 256],
        num_input_filters=128,
        num_anchor_per_loc=2,
        encode_background_as_zeros=True,
        use_direction_classifier=True,
        use_groupnorm=False,
        num_groups=32,
        use_bev=False,
        box_code_size=7,
        name="psa",
    ):
        """
        :param use_norm:
        :param num_class:
        :param layer_nums:
        :param layer_strides:
        :param num_filters:
        :param upsample_strides:
        :param num_upsample_filters:
        :param num_input_filters:
        :param num_anchor_per_loc:
        :param encode_background_as_zeros:
        :param use_direction_classifier:
        :param use_groupnorm:
        :param num_groups:
        :param use_bev:
        :param box_code_size:
        :param name:
        """
        super(PSA, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc  # 2
        self._use_direction_classifier = use_direction_classifier  # True
        self._use_bev = use_bev  # False
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:  # True
            if use_groupnorm:
                BatchNorm2d = change_default_args(num_groups=num_groups,
                                                  eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(
                    nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_input_filters,
                   num_filters[0],
                   3,
                   stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
            ),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(block2_input_filters,
                   num_filters[1],
                   3,
                   stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
            ),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
            ),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters),
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters),
                                          num_anchor_per_loc * 2, 1)

        self.bottle_conv = nn.Conv2d(sum(num_upsample_filters),
                                     sum(num_upsample_filters) // 3, 1)

        self.block1_dec2x = nn.MaxPool2d(kernel_size=2)  # C=64
        self.block1_dec4x = nn.MaxPool2d(kernel_size=4)  # C=64

        self.block2_dec2x = nn.MaxPool2d(kernel_size=2)  # C=128
        self.block2_inc2x = ConvTranspose2d(
            num_filters[1],
            num_filters[0] // 2,
            upsample_strides[1],
            stride=upsample_strides[1],
        )  # C=32

        self.block3_inc2x = ConvTranspose2d(
            num_filters[2],
            num_filters[1] // 2,
            upsample_strides[1],
            stride=upsample_strides[1],
        )  # C=64
        self.block3_inc4x = ConvTranspose2d(
            num_filters[2],
            num_filters[0] // 2,
            upsample_strides[2],
            stride=upsample_strides[2],
        )  # C=32

        self.fusion_block1 = nn.Conv2d(
            num_filters[0] + num_filters[0] // 2 + num_filters[0] // 2,
            num_filters[0],
            1,
        )
        self.fusion_block2 = nn.Conv2d(
            num_filters[0] + num_filters[1] + num_filters[1] // 2,
            num_filters[1], 1)
        self.fusion_block3 = nn.Conv2d(
            num_filters[0] + num_filters[1] + num_filters[2], num_filters[2],
            1)

        self.refine_up1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
            ),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.refine_up2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
            ),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.refine_up3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
            ),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

        C_Bottle = cfg.PSA.C_Bottle
        C = cfg.PSA.C_Reudce

        self.RF1 = Sequential(  # 3*3
            Conv2d(C_Bottle * 2, C, kernel_size=1, stride=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C,
                   C_Bottle * 2,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1),
            BatchNorm2d(C_Bottle * 2),
            nn.ReLU(inplace=True),
        )

        self.RF2 = Sequential(  # 5*5
            Conv2d(C_Bottle, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle, kernel_size=3, stride=1, padding=1,
                   dilation=1),
            BatchNorm2d(C_Bottle),
            nn.ReLU(inplace=True),
        )

        self.RF3 = Sequential(  # 7*7
            Conv2d(C_Bottle // 2, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C_Bottle // 2),
            nn.ReLU(inplace=True),
        )

        self.concat_conv1 = nn.Conv2d(num_filters[1],
                                      num_filters[1],
                                      kernel_size=3,
                                      padding=1)
        self.concat_conv2 = nn.Conv2d(num_filters[1],
                                      num_filters[1],
                                      kernel_size=3,
                                      padding=1)
        self.concat_conv3 = nn.Conv2d(num_filters[1],
                                      num_filters[1],
                                      kernel_size=3,
                                      padding=1)

        self.refine_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.refine_loc = nn.Conv2d(sum(num_upsample_filters),
                                    num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.refine_dir = nn.Conv2d(sum(num_upsample_filters),
                                        num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x1 = self.block1(x)
        up1 = self.deconv1(x1)

        x2 = self.block2(x1)
        up2 = self.deconv2(x2)
        x3 = self.block3(x2)
        up3 = self.deconv3(x3)
        coarse_feat = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(coarse_feat)
        cls_preds = self.conv_cls(coarse_feat)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(coarse_feat)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds

        blottle_conv = self.bottle_conv(coarse_feat)

        x1_dec2x = self.block1_dec2x(x1)
        x1_dec4x = self.block1_dec4x(x1)

        x2_dec2x = self.block2_dec2x(x2)
        x2_inc2x = self.block2_inc2x(x2)

        x3_inc2x = self.block3_inc2x(x3)
        x3_inc4x = self.block3_inc4x(x3)

        concat_block1 = torch.cat([x1, x2_inc2x, x3_inc4x], dim=1)
        fusion_block1 = self.fusion_block1(concat_block1)

        concat_block2 = torch.cat([x1_dec2x, x2, x3_inc2x], dim=1)
        fusion_block2 = self.fusion_block2(concat_block2)

        concat_block3 = torch.cat([x1_dec4x, x2_dec2x, x3], dim=1)
        fusion_block3 = self.fusion_block3(concat_block3)

        refine_up1 = self.RF3(fusion_block1)
        refine_up1 = self.refine_up1(refine_up1)
        refine_up2 = self.RF2(fusion_block2)
        refine_up2 = self.refine_up2(refine_up2)
        refine_up3 = self.RF1(fusion_block3)
        refine_up3 = self.refine_up3(refine_up3)

        branch1_sum_wise = refine_up1 + blottle_conv
        branch2_sum_wise = refine_up2 + blottle_conv
        branch3_sum_wise = refine_up3 + blottle_conv

        concat_conv1 = self.concat_conv1(branch1_sum_wise)
        concat_conv2 = self.concat_conv2(branch2_sum_wise)
        concat_conv3 = self.concat_conv3(branch3_sum_wise)

        PSA_output = torch.cat([concat_conv1, concat_conv2, concat_conv3],
                               dim=1)

        refine_cls_preds = self.refine_cls(PSA_output)
        refine_loc_preds = self.refine_loc(PSA_output)

        refine_loc_preds = refine_loc_preds.permute(0, 2, 3, 1).contiguous()
        refine_cls_preds = refine_cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict["Refine_loc_preds"] = refine_loc_preds
        ret_dict["Refine_cls_preds"] = refine_cls_preds

        if self._use_direction_classifier:
            refine_dir_preds = self.refine_dir(PSA_output)
            refine_dir_preds = refine_dir_preds.permute(0, 2, 3,
                                                        1).contiguous()
            ret_dict["Refine_dir_preds"] = refine_dir_preds

        return ret_dict
