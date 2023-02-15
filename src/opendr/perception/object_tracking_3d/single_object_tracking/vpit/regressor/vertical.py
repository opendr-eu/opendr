import torch
from torch import nn
import numpy as np
#from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.run import draw_pseudo_image


class Convolutional3kVerticalPositionRegressor(nn.Module):
    def __init__(
        self,
        input_filters,
        layer_strides=[2, 2, 2, 1],
        num_filters=[128, 64, 32, 1],
    ):
        super().__init__()

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

        result = torch.mean(features)
        return result


class ConvolutionalVerticalPositionRegressor(nn.Module):
    def __init__(
        self,
        input_filters,
        layer_strides=[1],
        num_filters=[1],
    ):
        super().__init__()

        layers = []
        layer_filters = []

        for i in range(len(num_filters)):
            layer_filters.append((num_filters[i-1] if i > 0 else input_filters, num_filters[i]))

        for i, (layer_stride, (in_filters, out_filters)) in enumerate(zip(layer_strides, layer_filters)):
            layers.append(
                nn.Conv2d(in_filters, out_filters, 1, stride=layer_stride)
            )
            if i < len(layer_filters) - 1:
                layers.append(nn.BatchNorm2d(out_filters))
                layers.append(nn.ReLU())

        self.body = nn.Sequential(*layers)

    def forward(self, x):

        features = self.body(x)

        result = torch.mean(features)
        return result


class CenterLinearVerticalPositionRegressor(nn.Module):
    def __init__(
        self,
        input_filters,
        num_filters=[1],
    ):
        super().__init__()

        layers = []
        layer_filters = []

        for i in range(len(num_filters)):
            layer_filters.append((num_filters[i-1] if i > 0 else input_filters, num_filters[i]))

        for i, (in_filters, out_filters) in enumerate(layer_filters):
            layers.append(
                nn.Linear(in_filters, out_filters)
            )
            if i < len(layer_filters) - 1:
                layers.append(nn.BatchNorm1d(out_filters))
                layers.append(nn.ReLU())

        self.body = nn.Sequential(*layers)

    def forward(self, x):

        center = np.array(x.shape[-2:]) // 2
        point = x[:, :, center[0], center[1]]

        features = self.body(point)
        result = torch.mean(features)
        return result


def create_vertical_position_regressor(input_filters, type="convolutional", **kwargs):
    if type == "center_linear":
        return CenterLinearVerticalPositionRegressor(input_filters, **kwargs)
    elif type == "convolutional":
        return ConvolutionalVerticalPositionRegressor(input_filters, **kwargs)
    elif type == "convolutional_3k":
        return Convolutional3kVerticalPositionRegressor(input_filters, **kwargs)
    else:
        raise ValueError()
