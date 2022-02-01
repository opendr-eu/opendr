import torch
from torch import nn
import json


class Config(object):
    def __init__(self, filename=None, data_dict=None):
        if filename is not None:
            self.loadfromJSON(filename)
        if data_dict is not None:
            self.fromJSON(data_dict)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJSON(self, data):
        # print data
        if isinstance(data, dict):
            self.__dict__ = data
        else:
            self.__dict__ = json.loads(data)

    def saveJSON(self, filename):
        with open(filename, "w") as f:
            f.write(self.toJSON())

    def loadfromJSON(self, filename):
        with open(filename, "r") as f:
            self.__dict__ = json.load(f)


class BoFConfig(Config):
    def __init__(self, data_dict=None, filename=None):
        super(BoFConfig, self).__init__(data_dict=data_dict, filename=filename)
        if filename is None and data_dict is None:
            self.type = "lbof"
            self.n_codewords = 64
            self.k = 1  # Conv2d kernel
            self.s = 1  # Conv2d stride
            self.p = 2  # AvgPool2d padding
            # self.g = 0.1
            # self.pooling_type = "avg"
            # self.normalization = "abs_sum"
            # self.join = "none"
            self.input_dim = 0
            # self.device = None
            # self.enabled = True


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()

    def forward(self, input):
        similarities = torch.abs(input)
        out = similarities / (torch.sum(similarities, dim=1, keepdim=True) + 1e-18)
        return out


class LBoF(nn.Module):
    def __init__(self, config):
        super(LBoF, self).__init__()
        self.conv = nn.Conv2d(
            config.input_dim, config.n_codewords, config.k, config.s, bias=False
        )
        self.norm = Normalization()
        self.pool = nn.AvgPool2d(config.p, stride=1)

    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.pool(out)
        return out
