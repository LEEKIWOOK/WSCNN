import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import init


class Flattening(nn.Module):
    def __init__(self):
        super(Flattening, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."

    def __init__(self, dim, dropout=0.1, max_len=43):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1 or classname.find("ConvTranspose1d") != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1 or classname.find("LayerNorm") != -1:
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.zeros_(m.bias)


# class Exponential(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.exp(x)


# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def forward(self, x):
#         return self.lambd(x)


# def deinterleave(x, size):
#     s = list(x.shape)
#     return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def interleave(x, size):
#     s = list(x.shape)
#     return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()

#     def forward(self, pred, actual):
#         return torch.sqrt(self.mse(pred, actual))


# class RMSLELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()

#     def forward(self, pred, actual):
#         return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

# def grl_hook(coeff):
#     def fun1(grad):
#         return -coeff*grad.clone()
#     return fun1

# class GradReverse(Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg()

# def grad_reverse(x, lambd):
#     return GradReverse.apply(x)

# def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
#     return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


# class Reparametering(nn.Module):
#     def __init__(self, feature_dim):
#         super(Reparametering, self).__init__()
#         self.feature_dim = feature_dim

#     def forward(self, x):
#         mu = x[:, :self.feature_dim]
#         std = F.softplus(x[:, self.feature_dim:] - 5, beta=1)
#         z = self.reparametrize_n(mu, std)
#         return mu, std, z

#     def reparametrize_n(self, mu, std):
#         eps = std.data.new(std.size()).normal_().cuda()
#         return mu + eps.detach() * std
