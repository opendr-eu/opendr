import math
import torch
import torch.nn as nn


def init_block(in_channels, out_channels, stride, activation=nn.PReLU):
    """Builds the first block of the MobileFaceNet"""
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation)
    )


class MobileFaceNet(nn.Module):
    """Implements modified MobileFaceNet from https://arxiv.org/abs/1804.07573"""

    def __init__(self, input_size=112, embedding_size=128, num_classes=1, width_multiplier=1., feature=True):
        super(MobileFaceNet, self).__init__()
        assert embedding_size > 0
        assert num_classes > 0
        assert width_multiplier > 0
        self.feature = feature
        self.input_size = input_size
        # Set up of inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        first_channel_num = 56
        last_channel_num = 512
        self.features = [init_block(3, first_channel_num, 2)]

        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(56))
        self.features.append(nn.PReLU())

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        size_h, size_w = MobileFaceNet.get_input_res(self)
        size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, outp_size=(size_h, size_w)))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, outp_size=(size_h, size_w)))
                in_channel_num = output_channel

        # 1x1 expand block
        self.features.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           nn.PReLU()))
        self.features = nn.Sequential(*self.features)

        # Depth-wise pooling
        k_size = (MobileFaceNet.get_input_res(self)[0] // 16, MobileFaceNet.get_input_res(self)[1] // 16)
        self.dw_pool = nn.Conv2d(last_channel_num, last_channel_num, k_size,
                                 groups=last_channel_num, bias=False)
        self.bn1 = nn.BatchNorm2d(last_channel_num)
        self.conv1_extra = nn.Conv2d(last_channel_num, embedding_size, 1, stride=1, padding=0, bias=False)

        self.init_weights()

    def forward(self, x):
        size = x.shape[0]
        x = self.features(x)
        x = self.bn1(self.dw_pool(x))
        x = self.conv1_extra(x)
        x = x.view(size, -1)
        return x

    @staticmethod
    def get_input_res(self):
        return self.input_size[0], self.input_size[1]

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class InvertedResidual(nn.Module):
    """Implementation of the modified Inverted residual block"""

    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, 8, nn.PReLU, outp_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)


def make_activation(activation):
    """Factory for activation functions"""
    if activation != nn.PReLU:
        return activation(inplace=True)

    return activation()


class SELayer(nn.Module):
    """Implementation of the Squeeze-Excitaion layer from https://arxiv.org/abs/1709.01507"""

    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        assert squeeze_ratio >= 1
        assert inplanes > 0
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = make_activation(activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class ScaleFilter(nn.Module):
    """Implementaion of the ScaleFilter regularizer"""

    def __init__(self, q):
        super(ScaleFilter, self).__init__()
        assert 0 < q < 1
        self.q = q

    def forward(self, x):
        if not self.training:
            return x

        scale_factors = 1. + self.q \
            - 2 * self.q * torch.rand(x.shape[1], 1, 1, dtype=torch.float32, requires_grad=False).to(
                x.device)
        return x * scale_factors
