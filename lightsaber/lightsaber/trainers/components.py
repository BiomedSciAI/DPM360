#!/usr/bin/env python
import torch as T
from torch import nn
from abc import abstractmethod, ABC
from argparse import ArgumentParser


class BaseModel(nn.Module, ABC):
    """Docstring for BaseModel. """

    def __init__(self):
        """TODO: to be defined. """
        super(BaseModel, self).__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this model
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-lr', '--learning_rate', default=0.02, type=float)
        parser.add_argument('-bs', '--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser

    def _get_packed_last_time(self, output, lengths=None):
        """
        Return last valid output from packed sequence

        ref: https://blog.nelsonliu.me/2018/01/25/extracting-last-timestep-outputs-from-pytorch-rnns/
        """
        if isinstance(output, T.nn.utils.rnn.PackedSequence):
            # Unpack, with batch_first=True.
            output, lengths = T.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        
        if lengths is None:
            if self.batch_first:
                last_output = output[:, -1, :]
            else:
                last_output = output[-1, :, :]
        else:
            # Extract the outputs for the last timestep of each example
            idx = (T.LongTensor(lengths) - 1).view(-1, 1).expand(
                len(lengths), output.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if output.is_cuda:
                idx = idx.cuda(output.data.get_device())
            # Shape: (batch_size, rnn_hidden_dim)
            last_output = output.gather(
                time_dimension, T.autograd.Variable(idx)).squeeze(time_dimension)
        return last_output


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


class ClassifierMixin(object):
    def get_logit(self):
        bias = self._kwargs.get('op_bias', False)
        return nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

