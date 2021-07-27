#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from torch.autograd import Variable
from lightsaber import constants as C
from lightsaber.trainers.components import BaseModel
from lightsaber.trainers.components import FFBlock, ClassifierMixin


class MLPBase(BaseModel):
    """Multi-Layer Perceptron"""

    def __init__(self, input_dim, output_dim, hidden_dim, 
                 n_layers, 
                 bias=True, 
                 dropout=None, 
                 batch_first=True, 
                 act='ReLU',
                 *args, **kwargs
                ):
        """
        Parameters
        ----------
        input_dim : TODO
        hidden_dim : TODO
        n_layers : TODO
        bias : TODO
        dropout : TODO, optional
        batch_first: TODO, optional
        act: TODO, optional
        """
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

        self._bias = bias
        self._dropout = dropout
        self._act = getattr(nn.modules.activation, act)
        self._args = args
        self._kwargs = kwargs

        assert self.n_layers > 0, "Number of layers should be atleast 1"

        self.build_model()

    def build_model(self):
        components = [FFBlock(self.input_dim, self.hidden_dim, bias=self._bias)]
        if self._dropout is not None:
            components.append(nn.Dropout(self._dropout))
        components.append(self._act())

        for l_idx in range(1, self.n_layers):
            components = [FFBlock(self.input_dim, self.hidden_dim, bias=self._bias)]
            if self._dropout is not None:
                components.append(nn.Dropout(self._dropout))
            components.append(self._act())
        self.model = nn.Sequential(*components)
        return

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class should not be used bare")


    def __repr__(self):
        return self.model.__repr__()


class MLPClassifier(MLPBase, ClassifierMixin):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 n_layers, 
                 bias=True, 
                 dropout=None, 
                 batch_first=True, 
                 act='ReLU',
                 *args, **kwargs
                ):
        """TODO: to be defined. """
        MLPBase.__init__(self, input_dim, output_dim, hidden_dim, 
                 n_layers, 
                 bias=bias, 
                 dropout=dropout, 
                 batch_first=batch_first, 
                 act=act,
                 *args, **kwargs
                )
        self.logit = self.get_logit()
        self.dropout = nn.Dropout(dropout)
        
        # initialize weights
        self.init_weights()
    
    def get_logit(self):
        bias = self._kwargs.get('op_bias', False)
        return nn.Linear(self.hidden_dim, self.output_dim, bias=bias)
    
    def init_weights(self):
        # RNN portion
        super().init_weights()

        nn.init.orthogonal_(self.logit.weight)
        op_bias = self._kwargs.get('op_bias', False)
        if op_bias:
            nn.init.zeros_(self.logit.bias)
        return

    def forward(self, data, lengths=None, hidden=None):
        is_packed = isinstance(data, T.nn.utils.rnn.PackedSequence)

        if is_packed:
            data, lengths = pad_packed_sequence(data, batch_first=self.batch_first)

        out = self.model.forward(data)
        logit = self.logit(self.dropout(out))
        
        if self.batch_first:
            logit.squeeze_(1)
        else:
            logit.squeeze_(0)
        
        hidden = None

        if is_packed:
            logit = pack_padded_sequence(logit, lengths, batch_first=self._batch_first)

        return (logit, hidden)
