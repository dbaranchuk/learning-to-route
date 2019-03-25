import numpy as np
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 .
    Shamelessly stolen from https://github.com/tkipf/pygcn/
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionBlock(nn.Module):
    def __init__(self, inp_size, hid_size, out_size=None, activation=nn.ELU(),
                 residual=False, normalize_hid=False, normalize_out=False):
        """ Graph convolution layer with some options """
        nn.Module.__init__(self)
        out_size = out_size or inp_size
        assert (out_size == inp_size) or not residual
        self.conv = GraphConvolution(inp_size, hid_size)
        if normalize_hid:
            self.hid_norm = nn.LayerNorm(hid_size)
        self.activation = activation
        self.dense = nn.Linear(hid_size, out_size)
        self.residual = residual
        if normalize_out:
            self.out_norm = nn.LayerNorm(out_size)

    def forward(self, inp, adj):
        hid = self.conv(inp, adj)
        if hasattr(self, 'hid_norm'):
            hid = self.hid_norm(hid)
        hid = self.activation(hid)
        hid = self.dense(hid)
        if self.residual:
            hid += inp
        if hasattr(self, 'out_norm'):
            hid = self.out_norm(hid)
        return hid


def logsoftmax_any(logits, mask, dim=-1, keepdim=False):
    """
    Computes softmax log-probability for ANY of references: log sum_i softmax(outcome_i)
    :param logits: n-dimensional tensor of softmax inputs
    :param mask: boolean mask on logits, 1 or 0. Computes logp of any of 1's in mask
    :param dim: probabilities add up to 1 along this dimension
    :param keepdim: if False, reduces `dim` dimension, if True - keeps 1-element dimension
    """
    numerator_logits = torch.where(mask, logits, torch.full_like(logits, -float('inf')))
    log_numerator = torch.logsumexp(numerator_logits, dim=dim, keepdim=keepdim)
    log_denominator = torch.logsumexp(logits, dim=dim, keepdim=keepdim)
    return log_numerator - log_denominator


def get_device_of(x):
    """ returns device name of tensor or module """
    if isinstance(x, torch.nn.Module):
        x = next(iter(x.state_dict().values()))
    return str(x.device)
