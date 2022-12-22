# Adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class PoseDecoder(nn.Module):
    def __init__(
        self,
        num_ch_encoder: np.ndarray,
        num_input_features: int,
        num_frames_to_predict_for: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_ch_encoder = num_ch_encoder
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        setattr(self, 'squeeze', nn.Conv2d(self.num_ch_encoder[-1], 256, 1))
        setattr(self, 'pose_0', nn.Conv2d(num_input_features * 256, 256, 3, 1, 1))
        setattr(self, 'pose_1', nn.Conv2d(256, 256, 3, 1, 1))
        setattr(self, 'pose_2', nn.Conv2d(256, 6 * num_frames_to_predict_for, 1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: Tensor) -> Tuple[Tensor, Tensor]:
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(getattr(self, 'squeeze')(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = getattr(self, f'pose_{i}')(out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axis_angle = out[..., :3]
        translation = out[..., 3:]
        return axis_angle, translation
