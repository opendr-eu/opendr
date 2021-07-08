# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch as t
from torch import nn


class QuadraticConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height=3, kernel_width=3, stride=1, dilation=1, q=1):
        super().__init__()
        in_channels *= q
        self.inChannels = in_channels
        self.outChannels = out_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.dilation = dilation
        self.weight_linear = nn.Parameter(t.Tensor(out_channels, in_channels, kernel_height, kernel_width))
        self.weight_volterra = nn.Parameter(
            t.Tensor(out_channels, in_channels, kernel_height * kernel_width, kernel_height * kernel_width))
        self.bias = nn.Parameter(t.Tensor(out_channels))
        self.q = q

        self.reset_parameters()

    @t.no_grad()
    def reset_parameters(self):
        bound = 0.01
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.xavier_uniform_(self.weight_linear)
        nn.init.xavier_uniform_(self.weight_volterra)
        # Values below diagonal are not needed for the Volterra kernel
        t.triu(input=self.weight_volterra, out=self.weight_volterra)

    def forward(self, image):
        # SelfONN Taylor series expansion
        image = t.cat([(image ** (i + 1)) for i in range(self.q)], dim=1)

        vertical_pad = int(math.ceil(self.kernel_height / 2)) - 1
        horizontal_pad = int(math.ceil(self.kernel_width / 2)) - 1

        batch_size, in_channels, in_h, in_w = image.shape
        out_height = int((in_h + 2 * vertical_pad - self.dilation * (self.kernel_height - 1) - 1) / self.stride + 1)
        out_width = int((in_w + 2 * horizontal_pad - self.dilation * (self.kernel_width - 1) - 1) / self.stride + 1)

        unfold_function = t.nn.Unfold(kernel_size=(self.kernel_height, self.kernel_width),
                                      dilation=self.dilation,
                                      padding=(vertical_pad, horizontal_pad),
                                      stride=self.stride)
        unfolded_input = unfold_function(image)

        unfolded_output_linear = unfolded_input.transpose(1, 2).matmul(
            self.weight_linear.view(self.weight_linear.shape[0], -1).t()).transpose(1,
                                                                                    2)
        xt = unfolded_input.transpose(1, 2)
        xt = xt.view(xt.shape[0], xt.shape[1], in_channels, -1, 1)
        x = xt.transpose(3, 4)
        xtx = t.matmul(xt, x)

        unfolded_output_volterra = xtx.view(xtx.shape[0], xtx.shape[1], -1).matmul(
            self.weight_volterra.view(self.weight_volterra.shape[0], -1).t()).transpose(1, 2)
        sum_of_orders = (unfolded_output_linear + unfolded_output_volterra).view(batch_size, self.outChannels,
                                                                                 out_height, out_width)
        return sum_of_orders + self.bias[None, :, None, None]


class QuadraticSelfOnnNet(nn.Module):
    def __init__(self, output_classes_n, q):
        super().__init__()
        self.convs = nn.Sequential(
            QuadraticConv(1, 20, q=q),
            nn.MaxPool2d(2),
            nn.Tanh(),
            QuadraticConv(20, 20, q=q),
            nn.MaxPool2d(2),
            nn.Tanh()
        )
        self.dropout = nn.Dropout2d(p=0.2)
        self.decoder = nn.Linear(800, output_classes_n)

    def forward(self, x):
        x = self.convs(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout(x)
        x = self.decoder(x)
        return t.log_softmax(x, dim=1)
