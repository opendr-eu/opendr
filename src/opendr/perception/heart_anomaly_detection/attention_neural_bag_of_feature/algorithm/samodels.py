import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_codeword, series_length, att_type):
        super(SelfAttention, self).__init__()

        assert att_type in ['spatialsa', 'temporalsa', 'spatiotemporal']

        self.att_type = att_type
        self.hidden_dim = 128

        self.n_codeword = n_codeword
        self.series_length = series_length

        if self.att_type == 'spatiotemporal':
            self.w_s = nn.Linear(n_codeword, self.hidden_dim)
            self.w_t = nn.Linear(series_length, self.hidden_dim)
        elif self.att_type == 'spatialsa':
            self.w_1 = nn.Linear(series_length, self.hidden_dim)
            self.w_2 = nn.Linear(series_length, self.hidden_dim)
        elif self.att_type == 'temporalsa':
            self.w_1 = nn.Linear(n_codeword, self.hidden_dim)
            self.w_2 = nn.Linear(n_codeword, self.hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.alpha = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

    def forward(self, x):
        # dimension order of x: batch_size, in_channels, series_length

        # clip the value of alpha to [0, 1]
        with torch.no_grad():
            self.alpha.copy_(torch.clip(self.alpha, 0.0, 1.0))

        if self.att_type == 'spatiotemporal':
            q = self.w_t(x)
            x_s = x.transpose(-1, -2)
            k = self.w_s(x_s)
            qkt = q @ k.transpose(-2, -1)*(self.hidden_dim**-0.5)
            mask = F.sigmoid(qkt)
            x = x * self.alpha + (1.0 - self.alpha) * x * mask

        elif self.att_type == 'temporalsa':
            x1 = x.transpose(-1, -2)
            q = self.w_1(x1)
            k = self.w_2(x1)
            mask = F.softmax(q @ k.transpose(-2, -1)*(self.hidden_dim**-0.5), dim=-1)
            mask = self.drop(mask)
            temp = mask @ x1
            x1 = x1 * self.alpha + (1.0 - self.alpha) * temp
            x = x1.transpose(-2, -1)

        elif self.att_type == 'spatialsa':
            q = self.w_1(x)
            k = self.w_2(x)
            mask = F.softmax(q @ k.transpose(-2, -1)*(self.hidden_dim**-0.5), dim=-1)
            mask = self.drop(mask)
            temp = mask @ x
            x = x * self.alpha + (1.0 - self.alpha) * temp

        return x
