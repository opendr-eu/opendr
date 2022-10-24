import torch
import numpy as np
from torch import nn
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.torchplus_tanet.tools import (
    change_default_args,
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.torchplus_tanet.nn import (
    Empty,
    GroupNorm,
    Sequential,
)
from opendr.engine.channel_pruning import (
    ChannelPruningBatchNorm2D,
    ChannelPruningConvolution2D,
    ChannelPruningConvolutionTranspose2D,
)


class PRPN(nn.Module):
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
        name="prpn",
    ):
        super(PRPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[: i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[: i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                raise ValueError()
                BatchNorm2d = change_default_args(num_groups=num_groups, eps=1e-3)(
                    GroupNorm
                )
            else:
                BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(
                    ChannelPruningBatchNorm2D
                )
            Conv2d = change_default_args(bias=False)(ChannelPruningConvolution2D)
            ConvTranspose2d = change_default_args(bias=False)(
                ChannelPruningConvolutionTranspose2D
            )
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(ChannelPruningConvolution2D)
            ConvTranspose2d = change_default_args(bias=True)(
                ChannelPruningConvolutionTranspose2D
            )

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

        b1_conv = Conv2d(num_input_filters, num_filters[0], 3, stride=layer_strides[0])
        b1_bnorm = BatchNorm2d(num_filters[0])

        b1_conv.add_link(b1_bnorm)

        self.block1 = Sequential(nn.ZeroPad2d(1), b1_conv, b1_bnorm, nn.ReLU(),)
        for i in range(layer_nums[0]):

            conv = Conv2d(num_filters[0], num_filters[0], 3, padding=1)
            b1_bnorm.add_link(conv)
            self.block1.add(conv)

            b1_bnorm = BatchNorm2d(num_filters[0])
            conv.add_link(b1_bnorm)

            self.block1.add(b1_bnorm)
            self.block1.add(nn.ReLU())

        b1_dconv = ConvTranspose2d(
            num_filters[0],
            num_upsample_filters[0],
            upsample_strides[0],
            stride=upsample_strides[0],
        )
        b1_dbnorm = BatchNorm2d(num_upsample_filters[0])

        b1_bnorm.add_link(b1_dconv)
        b1_dconv.add_link(b1_dbnorm)

        self.deconv1 = Sequential(b1_dconv, b1_dbnorm, nn.ReLU(),)

        b2_conv = Conv2d(
            block2_input_filters, num_filters[1], 3, stride=layer_strides[1],
        )
        b2_bnorm = BatchNorm2d(num_filters[1])

        b1_bnorm.add_link(b2_conv)
        b2_conv.add_link(b2_bnorm)

        self.block2 = Sequential(nn.ZeroPad2d(1), b2_conv, b2_bnorm, nn.ReLU(),)
        for i in range(layer_nums[1]):

            conv = Conv2d(num_filters[1], num_filters[1], 3, padding=1)
            b2_bnorm.add_link(conv)
            self.block2.add(conv)

            b2_bnorm = BatchNorm2d(num_filters[1])
            conv.add_link(b2_bnorm)
            self.block2.add(b2_bnorm)
            self.block2.add(nn.ReLU())

        b2_dconv = ConvTranspose2d(
            num_filters[1],
            num_upsample_filters[1],
            upsample_strides[1],
            stride=upsample_strides[1],
        )
        b2_dbnorm = BatchNorm2d(num_upsample_filters[1])

        b2_bnorm.add_link(b2_dconv)
        b2_dconv.add_link(b2_dbnorm)

        self.deconv2 = Sequential(b2_dconv, b2_dbnorm, nn.ReLU(),)

        b3_conv = Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2])
        b3_bnorm = BatchNorm2d(num_filters[2])

        b2_bnorm.add_link(b3_conv)
        b3_conv.add_link(b3_bnorm)

        self.block3 = Sequential(nn.ZeroPad2d(1), b3_conv, b3_bnorm, nn.ReLU(),)
        for i in range(layer_nums[2]):

            conv = Conv2d(num_filters[2], num_filters[2], 3, padding=1)
            b3_bnorm.add_link(conv)
            self.block3.add(conv)

            b3_bnorm = BatchNorm2d(num_filters[2])
            conv.add_link(b3_bnorm)

            self.block3.add(b3_bnorm)
            self.block3.add(nn.ReLU())

        b3_dconv = ConvTranspose2d(
            num_filters[2],
            num_upsample_filters[2],
            upsample_strides[2],
            stride=upsample_strides[2],
        )
        b3_dbnorm = BatchNorm2d(num_upsample_filters[2])

        self.deconv3 = Sequential(b3_dconv, b3_dbnorm, nn.ReLU(),)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.conv_cls = ChannelPruningConvolution2D(
            sum(num_upsample_filters), num_cls, 1
        )
        self.conv_box = ChannelPruningConvolution2D(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1
        )

        b1_dconv.add_link(self.conv_cls)
        b2_dconv.add_link(self.conv_cls)
        b3_dconv.add_link(self.conv_cls)

        b1_dconv.add_link(self.conv_box)
        b2_dconv.add_link(self.conv_box)
        b3_dconv.add_link(self.conv_box)

        if use_direction_classifier:
            self.conv_dir_cls = ChannelPruningConvolution2D(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1
            )
            b1_dconv.add_link(self.conv_dir_cls)
            b2_dconv.add_link(self.conv_dir_cls)
            b3_dconv.add_link(self.conv_dir_cls)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict
