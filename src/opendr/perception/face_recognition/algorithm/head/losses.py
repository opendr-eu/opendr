from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


# Support: ['ArcFace', 'CosFace', 'SphereFace', 'AMSoftmax', 'Classifier']

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:label
            in_features: size of each input sample
            out_features: size of each output sample
            device: the device on which the model is trained. Supports 'cuda' and 'cpu'
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)), 1e-9, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- Convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device == 'cuda':
            one_hot = one_hot.cuda(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device: the device on which the model is trained. Supports 'cuda' and 'cpu'
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- Convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device == 'cuda':
            one_hot = one_hot.cuda(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features = ' + str(self.in_features) \
            + ', out_features = ' + str(self.out_features) \
            + ', s = ' + str(self.s) \
            + ', m = ' + str(self.m) + ')'


class SphereFace(nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device: the device on which the model is trained. Supports 'cuda' and 'cpu'
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, device, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5.0
        self.iter = 0
        self.device = device

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        norm_of_feature = torch.norm(input, 2, 1)

        # --------------------------- Convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        if self.device == 'cuda':
            one_hot = one_hot.cuda(self.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features = ' + str(self.in_features) \
            + ', out_features = ' + str(self.out_features) \
            + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class AMSoftmax(nn.Module):
    r"""Implement of AMSoftmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device: the device on which the model is trained. Supports 'cuda' and 'cpu'
        m: margin
        s: scale of outputs
    """

    def __init__(self, in_features, out_features, device, m=0.35, s=30.0):
        super(AMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.device = device

        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # Initialize kernel

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # For numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.bool()
        output = cos_theta * 1.0
        output[index] = phi[index]  # Only change the correct predicted output
        output *= self.s  # Scale up in order to make softmax work, first introduced in normface

        return output


class Classifier(nn.Module):
    r"""Simple head for classifier mode:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device: the device on which the model is trained. Supports 'cuda' and 'cpu'
        """

    def __init__(self, in_features, out_features, device):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, int(in_features / 2))
        self.fc2 = nn.Linear(int(in_features / 2), out_features)
        self.device = device

    def forward(self, x, label=None):
        x = x.to(device=self.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
