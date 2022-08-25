#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as T
import torch.nn.functional as F
from torch.autograd import Variable

eps = 1e-6


# ******************************************************************
#                        Utility Functions
# TODO
# * possibly extend gumbel softmax to return top k . https://bit.ly/2GSzC8p
# 
# *******************************************************************
def gumbel_softmax_nd(X, axis=1, hard=True, tau=1.):
    """
    Gumbel_softmax on an axis. support for n-dim gumbel-softmax
    Default functional takes the softmax along the last axis

    Parameters
    ----------
    X: input logits
    axis: axis to take the softmax along.  (Default=1)
    hard: True/False. Hard employs ST trick, while soft leads to reparameterization trick.
    tau: temperature for softmax.
    """
    X_size = X.size()

    # switching last dim of X with required axis
    trans_X = X.transpose(axis, len(X_size) - 1)
    trans_size = trans_X.size()
   
    # Flattening other axes as (\sum_{all axis} \times softmax_dim)
    X_2d = trans_X.contiguous().view(-1, trans_size[-1])
    gsoft_max_2d = F.gumbel_softmax(X_2d, hard=hard, tau=tau)

    gsoft_max_2d = gsoft_max_2d.view(*trans_size)
    return gsoft_max_2d.transpose(axis, len(X_size) - 1)
 

def softmax_nd(X, axis=1):
    """Softmax on an axis. n-dimensional softmax
    ref: https://github.com/ixaxaar/pytorch-dnc/blob/master/dnc/util.py#L72

    Softmax on an axis

    Arguments:
      X {Tensor} -- input Tensor

    Keyword Arguments:
      axis {number} -- axis on which to take softmax on (default: {1})

    Returns:
      Tensor -- Softmax output Tensor
    """
    X_size = X.size()

    trans_X = X.transpose(axis, len(X_size) - 1)
    trans_size = trans_X.size()

    X_2d = trans_X.contiguous().view(-1, trans_size[-1])
    if '0.3' in T.__version__:
        soft_max_2d = F.softmax(X_2d, -1)
    else:
        soft_max_2d = F.softmax(X_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(X_size) - 1)


def batch_cosine(a, b, dimA=2, dimB=2, normBy=2):
    """Batchwise Cosine distance
    ref: https://github.com/ixaxaar/pytorch-dnc/blob/master/dnc/util.py#L47

    Cosine distance

    Arguments:
        a {Tensor} -- A 3D Tensor (b * m * w)
        b {Tensor} -- A 3D Tensor (b * r * w)

    Keyword Arguments:
        normBy {number} -- order of norm (default: {2})
        dimA {number} -- exponent value of the norm for `a` (default: {2})
        dimB {number} -- exponent value of the norm for `b` (default: {1})

    Returns:
        Tensor -- Batchwise cosine distance (b * r * m)
    """
    a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + eps
    b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + eps

    x = T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
        T.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + eps)
    # apply_dict(locals())
    return x


def last_padded_data(output, lengths, batch_first=True):
    """
    https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
    """
    idx = (T.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), output.size(2))  # bs * N_dim, each row contains max_time_len

    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    # Shape: (batch_size, rnn_hidden_dim)
    last_output = output.gather(
        time_dimension, Variable(idx)).squeeze(time_dimension)
    return last_output
