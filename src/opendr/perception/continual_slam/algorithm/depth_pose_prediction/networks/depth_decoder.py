# Adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import Conv3x3, ConvBlock


class DepthDecoder(nn.Module):
    def __init__(
            self,
            num_ch_encoder: np.ndarray,
            scales: Tuple[int, ...] = (0, 1, 2, 3),
            use_skips: bool = True,
    ) -> None:
        super().__init__()

        self.scales = scales
        self.use_skips = use_skips
        self.num_output_channels = 1

        self.num_ch_encoder = num_ch_encoder
        self.num_ch_decoder = np.array([16, 32, 64, 128, 256])

        self.convs = {}
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_encoder[-1] if i == 4 else self.num_ch_decoder[i + 1]
            num_ch_out = self.num_ch_decoder[i]
            setattr(self, f'upconv_{i}_0', ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_decoder[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_encoder[i - 1]
            num_ch_out = self.num_ch_decoder[i]
            setattr(self, f'upconv_{i}_1', ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            setattr(self, f'dispconv_{s}', Conv3x3(self.num_ch_decoder[s],
                                                   self.num_output_channels))

        self.sigmoid = nn.Sigmoid()
        self.output = {}

    def forward(self, input_features: Tensor) -> Dict[Tuple[str, int], Tensor]:
        self.output = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = getattr(self, f'upconv_{i}_0')(x)
            if self.use_skips and i > 0:
                # Difference to monodepth2 implementation to deal with image resolutions
                # that cannot be integer-divided by 2, 4, 8, etc.
                # Only required when evaluating the depth
                x = [F.interpolate(x, size=input_features[i - 1].shape[2:], mode='nearest')]
                x += [input_features[i - 1]]
            else:
                x = [F.interpolate(x, scale_factor=2, mode='nearest')]
            x = torch.cat(x, 1)
            x = getattr(self, f'upconv_{i}_1')(x)
            if i in self.scales:
                # monodepth2 paper (w/o uncertainty)
                self.output[('disp', i)] = self.sigmoid(getattr(self, f'dispconv_{i}')(x))

        return self.output
