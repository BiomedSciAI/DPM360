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


class RNNBase(BaseModel):
    def __init__(self, 
        input_dim, output_dim, hidden_dim,
        rnn_class=C.PYTORCH_CLASS_DICT['LSTM'],
        n_layers=1, 
        bias=True,
        dropout=0, 
        recurrent_dropout=0.,
        batch_first=True, 
        bidirectional=False, 
        *args, **kwargs):
        """TODO: to be defined. """
        BaseModel.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn_class = rnn_class
        self.n_layers = n_layers
        self.bias = bias
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Forward compatibility
        self._args = args
        self._kwargs = kwargs

        self.rnn = self.get_rnn()
        self.dropout = nn.Dropout(self.dropout)
    
    def get_rnn(self):
        rnn = self.rnn_class(self.input_dim, self.hidden_dim, self.n_layers,
                             bias=self.bias, 
                             dropout=self.recurrent_dropout,
                             batch_first=self.batch_first, 
                             bidirectional=self.bidirectional)
        return rnn
    
    def init_weights(self):
        # RNN portion
        if self.rnn_class == C.PYTORCH_CLASS_DICT['LSTM']:
            """Use orthogonal init for recurrent layers, xavier uniform for input layers
            Bias is 0 except for forget gate
            """
            unit_forget_bias = self._kwargs.get('unit_forget_bias', True)
            for name, param in self.rnn.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name and unit_forget_bias:
                    nn.init.zeros_(param.data)
                    param.data[self.hidden_dim:2 * self.hidden_dim] = 1
        elif self.rnn_class == C.PYTORCH_CLASS_DICT['GRU']:
            """Use orthogonal init for recurrent layers, xavier uniform for input layers
            Bias is 0 except for forget gate
            """
            unit_forget_bias = self._kwargs.get('unit_forget_bias', True)
            for name, param in self.rnn.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name and unit_forget_bias:
                    nn.init.zeros_(param.data)
        else:
            log.warning('weights not initialized')
            pass
        return
    
    def init_hidden(self, batch_size, device='cpu'):
        # RNN takes hidden as (n_layers * n_dir, bs, hs)
        n_dir = 1 if not self.bidirectional else 2
        if self.rnn_class == C.PYTORCH_CLASS_DICT['LSTM']:
            h_0 = T.rand(self.n_layers * n_dir, batch_size, self.hidden_dim).to(device)  
            c_0 = T.rand(self.n_layers * n_dir, batch_size, self.hidden_dim).to(device)
            hidden = (h_0, c_0)
        elif self.rnn_class == C.PYTORCH_CLASS_DICT['GRU']:
            h_0 = T.rand(self.n_layers * n_dir, batch_size, self.hidden_dim).to(device)  
            hidden = h_0
        else:
            hidden = None
        return hidden
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class should not be used bare")
        
    
class RNNClassifier(RNNBase):
    def __init__(self, 
                 input_dim, output_dim, hidden_dim,
                 rnn_class=C.PYTORCH_CLASS_DICT['LSTM'],
                 n_layers=1, bias=True, dropout=0, 
                 recurrent_dropout=0.,
                 batch_first=True, 
                 bidirectional=False, 
                 *args, **kwargs):
        """TODO: to be defined. """
        RNNBase.__init__(self, 
            input_dim, output_dim, hidden_dim,
            rnn_class=rnn_class,
            n_layers=n_layers, bias=bias, dropout=dropout, 
            recurrent_dropout=recurrent_dropout,
            batch_first=batch_first, 
            bidirectional=bidirectional, 
            *args, **kwargs)
        self.logit = self.get_logit()
        
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
        if isinstance(data, T.nn.utils.rnn.PackedSequence):
            _is_packed = True
            packed_data = data
            bs = packed_data.batch_sizes.max()
        else:
            bs = data.shape[0] if self.batch_first else data.shape[1]
            _is_packed = False
            if lengths is None:
                # Assuming equal lengths for data
                packed_data = data
            else:
                packed_data = pack_padded_sequence(Variable(data), lengths, batch_first=self.batch_first)
                assert bs == packed_data.batch_sizes.max()

        if hidden is None:
            hidden = self.init_hidden(bs, device=data.device)

        out, hidden = self.rnn.forward(packed_data, hidden)
        out = self._get_packed_last_time(out)
        logit = self.logit(self.dropout(out))
        return logit, hidden
