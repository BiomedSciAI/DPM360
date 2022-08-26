#!/usr/bin/env python
import torch as T
from torch import nn
from abc import abstractmethod, ABC
from argparse import ArgumentParser


class FFBlock(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.ones_(self.bias)
            #  nn.init.zeros_(self.bias)
        return


class ResidualBlock(nn.Module):
    def __init__(self, model, act='ReLU'):
        self.model = model
        self.act = getattr(nn, act)()

    def forward(self, x, hx=None, lengths=None):
        #  import ipdb; ipdb.set_trace()  # BREAKPOINT
        op, hx = self.model(x, hx, lengths)
        op = op + x
        op = self.act(op)
        return op, hx

