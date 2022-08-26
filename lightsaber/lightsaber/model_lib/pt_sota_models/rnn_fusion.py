#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from lightsaber.trainers.pt_trainer import BaseModel
import math


class RNN(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim, summary_dim=None,
                 rnn_class=nn.LSTM,
                 n_layers=1, bias=True, dropout=0, recurrent_dropout=0.,
                 batch_first=True, bidirectional=False, 
                 *args, **kwargs):
        """TODO: to be defined. """
        BaseModel.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.summary_dim = summary_dim
        self.summary_n_layers = kwargs.get('summary_n_layers', 1)
        self.summary_hidden_dim = kwargs.get('summary_hidden_dim', self.hidden_dim)

        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout

        self.batch_first = batch_first
        self.bidirection = bidirectional

        # Forward compatibility
        self._args = args
        self._kwargs = kwargs

        self.rnn = rnn_class(input_dim, hidden_dim, n_layers,
                             bias=bias, dropout=recurrent_dropout,
                             batch_first=self.batch_first, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        if self.summary_dim is None or self.summary_dim == 0:
            self.logit = nn.Linear(hidden_dim, output_dim)

        else:
            _input_dim = self.hidden_dim + self.summary_dim
            if self.summary_n_layers < 0:
                raise ValueError('at least 1 requried')
            elif self.summary_n_layers >= 1:
                _intermediate_dim = self.summary_hidden_dim
                _logit = [nn.Linear(_input_dim, _intermediate_dim), nn.ReLU()]
                for l in range(self.summary_n_layers - 1):
                    _logit += [nn.Linear(_intermediate_dim, _intermediate_dim), nn.ReLU()]
            else:
                _intermediate_dim = _input_dim
                _logit = []
            _logit.append(nn.Linear(_intermediate_dim, self.output_dim))
            self.logit = nn.Sequential(*_logit)

    def forward(self, data, lengths, summary=None, hidden=None):
        packed_data = pack_padded_sequence(Variable(data), lengths, batch_first=self.batch_first)
        out, hidden = self.rnn.forward(packed_data, hidden)
        out = self._get_packed_last_time(out)
        if summary is not None:
            out = out / 2 + 0.5  # tanh to sigmoig
            out = T.cat([out, summary], dim=1)
    
        logit = self.logit(self.dropout(out))

        return logit, hidden
    

class EmbeddingRNN(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim=None, 
                 summary_dim=None,
                 embedding_class=nn.Linear, rnn_class=nn.LSTM,
                 n_layers=1, bias=True, dropout=0, recurrent_dropout=0.,
                 batch_first=True, bidirectional=False, 
                 *args, **kwargs):
        BaseModel.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        if self.embedding_dim is not None:
            self.embedding_layer = embedding_class(input_dim, self.embedding_dim)
        else:
            self.embedding_layer = nn.Identity
            
        self.rnn = RNN(embedding_dim, output_dim, hidden_dim, summary_dim=summary_dim, rnn_class=rnn_class,
                       n_layers=n_layers, bias=bias, dropout=dropout, recurrent_dropout=recurrent_dropout,
                       batch_first=batch_first, bidirectional=bidirectional, 
                       *args, **kwargs)
        return
    
    def forward(self, data, lengths, summary=None, hidden=None):
        emb = self.embedding_layer(data)
        logit, hidden = self.rnn(emb, lengths, hidden=hidden)
        return logit, hidden


RNNS = ['LSTM', 'GRU']


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 n_layers=1, bias=True, recurrent_dropout=0.,
                 batch_first=True,
                 bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.batch_first = batch_first

        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type)

        self.rnn = rnn_cell(input_dim, hidden_dim, n_layers, 
                            bias=bias, dropout=recurrent_dropout,
                            batch_first=self.batch_first, bidirectional=self.bidirectional)

    def forward(self, data, lengths, hidden=None):
        packed_data = pack_padded_sequence(Variable(data), lengths, batch_first=self.batch_first)
        packed_output, hidden = self.rnn(packed_data, hidden)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=self.batch_first)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, batch_first=True):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.query_dim = query_dim
        self.batch_first = batch_first

    def forward(self, query, keys, values):
        # if batch_first = True:
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[Bx1xT], lin_comb:[BxV]

        query.unsqueeze_(1) # [BxQ] -> [Bx1xQ]

        if not self.batch_first:
            keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        else:
            keys = keys.transpose(1,2) #[BxTxK] -> [BxKxT]
        energy = T.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        if not self.batch_first:
            values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = T.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination
    
    def __repr__(self):
        string = f"""Attention(query_dim={self.query_dim}, scale={self.scale}, batch_first={self.batch_first})
        """
        return string

    
class RNN_Attention(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim,
                 summary_dim=None,
                 n_layers=1, 
                 dropout=0.,
                 recurrent_dropout=0.,
                 batch_first=True,
                 bias=True,
                 bidirectional=True, rnn_type='GRU',
                 *args, **kwargs):
        BaseModel.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.summary_dim = summary_dim

        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.summary_n_layers = kwargs.get('summary_n_layers', 1)
        self.summary_hidden_dim = kwargs.get('summary_hidden_dim', self.hidden_dim)
        
        self.encoder = Encoder(self.input_dim, self.hidden_dim, 
                               n_layers=n_layers, 
                               bias=bias,
                               recurrent_dropout=recurrent_dropout,
                               batch_first=batch_first,
                               bidirectional=bidirectional,
                               rnn_type=rnn_type)
        
        att_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.att_dim = att_dim
        self.attention = Attention(att_dim, att_dim, att_dim, batch_first=self.batch_first)
        
        if self.summary_dim is None or self.summary_dim == 0:
            self.decoder = nn.Linear(att_dim, self.output_dim)

        else:
            _input_dim = self.att_dim + self.summary_dim
            if self.summary_n_layers < 0:
                raise ValueError('at least 1 requried')
            elif self.summary_n_layers >= 1:
                _intermediate_dim = self.summary_hidden_dim
                _logit = [nn.Linear(_input_dim, _intermediate_dim), nn.ReLU()]
                for l in range(self.summary_n_layers - 1):
                    _logit += [nn.Linear(_intermediate_dim, _intermediate_dim), nn.ReLU()]
            else:
                _intermediate_dim = _input_dim
                _logit = []
            _logit.append(nn.Linear(_intermediate_dim, self.output_dim))
            self.decoder = nn.Sequential(*_logit)

        self.dropout = nn.Dropout(dropout)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input_data, length, summary=None, hidden=None):
        outputs, hidden = self.encoder(input_data, length, hidden)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = T.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        energy, linear_combination = self.attention(hidden, outputs, outputs) 
        
        if summary is not None:
            linear_combination = F.sigmoid(linear_combination) # / 2 + 0.5  # tanh to sigmoig
            linear_combination = T.cat([linear_combination, summary], dim=1)
            
        logits = self.decoder(self.dropout(linear_combination))
        return logits, energy



class RNN_Attention_latefusion(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim, summary_dim,
                 n_layers=1, 
                 dropout=0.,
                 recurrent_dropout=0.,
                 batch_first=True,
                 bias=True,
                 bidirectional=True, rnn_type='GRU',
                 *args, **kwargs):
        BaseModel.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.summary_dim = summary_dim

        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.summary_n_layers = kwargs.get('summary_n_layers', 1)
        self.summary_hidden_dim = kwargs.get('summary_hidden_dim', self.hidden_dim)
        
        self.encoder = Encoder(self.input_dim, self.hidden_dim, 
                               n_layers=n_layers, 
                               bias=bias,
                               recurrent_dropout=recurrent_dropout,
                               batch_first=batch_first,
                               bidirectional=bidirectional,
                               rnn_type=rnn_type)
        
        att_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.att_dim = att_dim
        self.attention = Attention(att_dim, att_dim, att_dim, batch_first=self.batch_first)
        

        # mlp for summary feature
        _input_dim = self.summary_dim
        if self.summary_n_layers <= 0:
            raise ValueError('at least 1 requried')
        elif self.summary_n_layers >= 1:
            _intermediate_dim = self.summary_hidden_dim
            _logit = [nn.Linear(_input_dim, _intermediate_dim), nn.ReLU()]
            for l in range(self.summary_n_layers - 1):
                _logit += [nn.Linear(_intermediate_dim, _intermediate_dim), nn.ReLU()]
        else:
            _intermediate_dim = _input_dim
            _logit = []
        self.mlp = nn.Sequential(*_logit)

        # decoder for concatenation of rnn output and mlp output
        self.decoder = nn.Linear(att_dim+self.summary_hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input_data, length, summary, hidden=None):
        outputs, hidden = self.encoder(input_data, length, hidden)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = T.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        energy, linear_combination = self.attention(hidden, outputs, outputs) 
        
        summary_output = self.mlp(summary)

        linear_combination = T.cat([linear_combination, summary_output], dim=1)
            
        logits = self.decoder(self.dropout(linear_combination))
        return logits, energy
