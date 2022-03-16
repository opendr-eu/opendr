import torch
from torch import nn

from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.run import draw_pseudo_image


class VerticalPositionRegressor(nn.Module):
    def __init__(
        self,
        input_filters,
        layer_strides=[2, 2, 2, 1],
        num_filters=[128, 64, 32, 1],
    ):
        super(VerticalPositionRegressor, self).__init__()

        layers = []
        layer_filters = []

        for i in range(len(num_filters)):
            layer_filters.append((num_filters[i-1] if i > 0 else input_filters, num_filters[i]))

        for i, (layer_stride, (in_filters, out_filters)) in enumerate(zip(layer_strides, layer_filters)):
            layers.append(
                nn.Conv2d(in_filters, out_filters, 3, padding=1, stride=layer_stride)
            )
            if i < len(layer_filters) - 1:
                layers.append(nn.BatchNorm2d(out_filters))
                layers.append(nn.ReLU())

        self.body = nn.Sequential(*layers)

    def forward(self, x):

        features = self.body(x)

        draw_pseudo_image(x[0], "./plots/train/vertical_in.png")
        draw_pseudo_image(features[0], "./plots/train/vertical.png")

        result = torch.mean(features)
        return result
