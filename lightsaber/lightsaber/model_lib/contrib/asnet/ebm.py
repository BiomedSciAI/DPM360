#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main module file for Explicit(E)-Blurred(B) memory augmented RNN

Contents of EB-RNN:

* RNN controller
* expicit memory
* blurred memory
* write gate (blur)
* read gate
"""

from __future__ import print_function

#  import os
#  import pandas as pd
import numpy as np

import torch as T
from torch import nn
#  from torch.autograd import Variable
#  from torch.nn.modules import rnn

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

#  from torch.nn.init import orthogonal, xavier_uniform

from .memory import ExplicitMem, BlurredMem, MixGate

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
log = logging.getLogger()


class EBmRNN(nn.Module):

    def __init__(self, input_size, hidden_size, cell_size=10, n_slots=5,
                 n_reads=2, tau=0.3, ctl_rnn='GRU', ctl_n_layers=2,
                 ctl_nonlinearity='tanh', clip=20, dropout=0.4, bias=True,
                 batch_first=True):
        """
        Explicit-Blurred Memory Augmented RNN. 

        Parameters
        ----------
        input_size
        hidden_size
        cell_size
        n_slots
        n_reads
        tau
        ctl_rnn
        ctl_n_layers
        ctl_nonlinearity
        clip
        dropout
        bias
        batch_first

        Example
        -------        
        """
        nn.Module.__init__(self)

        # Placeholders for dimensions
        self.input_size = input_size       # D: input vec dimension 
        self.hidden_size = hidden_size     # H: hidden size
        self.cell_size = cell_size         # W: dimension of memory
        self.n_slots = n_slots             # N: number of memory cells
        self.n_reads = n_reads             # r: reads
        self.tau = tau                     # tau: softmax temperature
        
        # Controller specification
        self.ctl_rnn = ctl_rnn
        self.ctl_n_layers = ctl_n_layers
        self.ctl_nonlinearity = ctl_nonlinearity

        # Overall parameters
        self.clip = clip
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first
        # --------------

        self.build_model()

        return

    def build_model(self):
        # Computed placeholders
        self.read_size = self.n_reads * self.cell_size  # r vectors of dim W
        self.output_size = self.ctl_n_layers * self.hidden_size + self.read_size  # op dim same as hidden dim + read dimensio
        self.erase_size = self.cell_size                # erase vec of size W 

        self.ctl_input_size = self.input_size + self.read_size
        self.ctl_output_size = (2 * self.read_size  # keys for two mem banks
                                + self.cell_size    # for candidate memory
                                + self.erase_size   # for erase vector
                                )

        if self.ctl_rnn == 'GRU':
            self.controller = nn.GRU(self.ctl_input_size, self.hidden_size, self.ctl_n_layers,
                                     bias=self.bias, batch_first=self.batch_first,
                                     dropout=self.dropout)
        elif self.ctl_rnn == 'LSTM':
            self.controller = nn.LSTM(self.ctl_input_size, self.hidden_size, self.ctl_n_layers,
                                      bias=self.bias, batch_first=self.batch_first,
                                      dropout=self.dropout)
        else:
            raise ValueError('Controller type not known. only GRU/LSTM supported')
       
        self.interface = nn.Sequential(nn.Linear(self.hidden_size, self.ctl_output_size),
                                       nn.ReLU())

        self.explicit = ExplicitMem(self.cell_size, self.n_slots, self.n_reads, 
                                    self.hidden_size, self.tau)
        self.blurred = BlurredMem(self.cell_size, self.n_slots, self.n_reads)
    
        self.out_gate = MixGate(self.cell_size)

        nn.init.orthogonal_(self.interface[0].weight)
        pass

    def init_hidden(self, bs, hx=None):
        """Resetting hidden states for each new batch.

        Passing batch_size allows model to be run with variable batch size. (?)
        """
        if hx is None:
            hx = (None, None, None, None, None)
        (chx, em, bm, em_index, last_read) = hx

        if chx is None:
            h = T.empty(self.ctl_n_layers, bs, self.hidden_size)   # n_layers * bs * H
            nn.init.xavier_uniform_(h)

            if self.ctl_rnn == 'GRU':
                chx = h
            elif self.ctl_rnn == 'LSTM':
                # FIXME: initialize each LSTM hidden dimension separately
                chx = (h, h.clone())

        if em is None:
            em = self.explicit.init_hidden(bs)

        if bm is None:
            bm = self.blurred.init_hidden(bs)

        if em_index is None:
            em_index = np.zeros([bs, self.n_slots])               # bs * N
            em_index.fill_(np.nan)

        if last_read is None:
            last_read = T.zeros(bs, self.read_size)  # bs * (RW)
        
        return chx, em, bm, em_index, last_read

    def _forward_once(self, ctl_input, hx, seq_idx):
        """
        # TODO: check dimensions
        ctl_input: bs * 1 * feature_dim
        hx:
            chx: n_layers * bs * H
            em['memory']: bs * W * N
            bm['memory']: bs * W * N
            em_index: bs * N
        seq_idx: int
        """

        (chx, em, bm, em_index) = hx
        _out, chx = self.controller(ctl_input.unsqueeze(1), chx)
        _out = _out.squeeze(1)                                   # bs * H

        _interface = self.interface(_out)                        # bs * (E)

        if self.clip != 0:
            out = T.clamp(_interface, -self.clip, self.clip)
        else:
            out = _interface

        # Interface vector
        k_em = out[:, :self.read_size]                                         # bs * RW
        k_bm = out[:, self.read_size:2 * self.read_size]                       # bs * RW
        m_t = out[:, 2 * self.read_size: 2 * self.read_size + self.cell_size]  # bs * W
        e_t = out[:, 2 * self.read_size + self.cell_size:]                     # bs * W

        #  print(m_t.shape)
        em_read, em, em_erased, em_index = self.explicit(k_em, em, m_t, chx, 
                                                         em_index, seq_idx)
        #  print(m_t.shape)
        bm_read, bm = self.blurred(k_bm, bm, m_t, em_erased, e_t)

        return (out, em_read, bm_read), (chx, em, bm, em_index) 

    def forward(self, input, lengths=None, hx=(None, None, None, None, None), save_trace=False):
        # is_packed = type(input) is PackedSequence
        # if is_packed:
        #     input, lengths = pad(input, batch_first=self.batch_first)
        #     max_length = lengths.numpy()[0]  # assuming sorted packed
        # else:
        #     max_length = input.size(1) if self.batch_first else input.size(0)
        #     lengths = [max_length] * input.size(0) if self.batch_first \
        #         else [max_length] * input.size(1)
        
        max_length = input.size(1) if self.batch_first else input.size(0)
        batch_size = input.size(0) if self.batch_first else input.size(1)

        #  print(batch_size, max_length)
        if not self.batch_first:
            input = input.transpose(0, 1)  # making it batch first

        (ctl_hidden, exp_hidden, blr_hidden, 
         em_index, l_read) = self.init_hidden(batch_size, hx)
 
        outs = [None] * max_length
        read_vecs = [None] * max_length                      # flattenned read vector

        all_mem = [None] * max_length                        # read weight vecs
        mix_vecs = [None] * max_length

        # Recurrent lopp
        for time in range(max_length):
            ctl_input = T.cat([input[:, time, :], l_read], 1)  # bs * feature_dim

            #  import ipdb; ipdb.set_trace()  # BREAKPOINT
            chx, em, bm, m_idx = ctl_hidden, exp_hidden, blr_hidden, em_index
            (_out, em_read, bm_read), (chx, em, bm, m_idx) =\
                self._forward_once(ctl_input, (chx, em, bm, m_idx), time)
            ctl_hidden = chx
            exp_hidden = em
            blr_hidden = bm
            em_index = m_idx
            
            #  outs[time] = _out

            # Computing gated read vector
            read_vecs[time], mix_vecs[time] = self.out_gate(em_read,  # bs * R * W
                                                            bm_read   # bs * R * W
                                                            )  # bs * R * 1, bs * R * W
            l_read = read_vecs[time].view(batch_size, -1)  # bs * RW
            outs[time] = T.cat([chx.transpose(0, 1).contiguous().view(batch_size, -1),
                                l_read], 1)

            #  print(em_index)

            if save_trace:
                all_mem[time] = (self.explicit.detach(em),
                                 self.blurred.detach(bm),
                                 mix_vecs[time].detach(),
                                 m_idx)
        # ------- end recursion

        outs = T.stack(outs)
        if self.batch_first:
            outs.transpose_(0, 1)

        if is_packed:
            outs = pack(outs, lengths, batch_first=self.batch_first)

        return outs, (ctl_hidden, exp_hidden, blr_hidden, em_index, l_read), all_mem

    def __repr__(self):
        """
        TODO: write a more detailed `repr` method
        """
        return super(EBmRNN, self).__repr__()


class EmRNN(nn.Module):

    def __init__(self, input_size, hidden_size, cell_size=10, n_slots=5,
                 n_reads=2, tau=0.3, ctl_rnn='GRU', ctl_n_layers=2,
                 ctl_nonlinearity='tanh', clip=20, dropout=0.4, bias=True,
                 batch_first=True):
        """
        Explicit-only Memory Augmented RNN. 

        Parameters
        ----------
        input_size
        hidden_size
        cell_size
        n_slots
        n_reads
        tau
        ctl_rnn
        ctl_n_layers
        ctl_nonlinearity
        clip
        dropout
        bias
        batch_first

        Example
        -------        
        """
        nn.Module.__init__(self)

        # Placeholders for dimensions
        self.input_size = input_size       # D: input vec dimension 
        self.hidden_size = hidden_size     # H: hidden size
        self.cell_size = cell_size         # W: dimension of memory
        self.n_slots = n_slots             # N: number of memory cells
        self.n_reads = n_reads             # r: reads
        self.tau = tau                     # tau: softmax temperature
        
        # Controller specification
        self.ctl_rnn = ctl_rnn
        self.ctl_n_layers = ctl_n_layers
        self.ctl_nonlinearity = ctl_nonlinearity

        # Overall parameters
        self.clip = clip
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first
        # --------------

        # Computed placeholders
        self.read_size = self.n_reads * self.cell_size  # r vectors of dim W
        self.output_size = self.ctl_n_layers * self.hidden_size + self.read_size  # op dim same as hidden dim + read dimensio
        #  self.erase_size = self.cell_size                # erase vec of size W 
        self.erase_size = 0  # only explicit memory

        self.ctl_input_size = self.input_size + self.read_size
        self.ctl_output_size = (1 * self.read_size  # keys for only exp mem banks
                                + self.cell_size    # for candidate memory
                                + self.erase_size   # for erase vector
                                )

        if self.ctl_rnn == 'GRU':
            self.controller = nn.GRU(self.ctl_input_size, self.hidden_size, self.ctl_n_layers,
                                     bias=self.bias, batch_first=self.batch_first,
                                     dropout=self.dropout)
        elif self.ctl_rnn == 'LSTM':
            self.controller = nn.LSTM(self.ctl_input_size, self.hidden_size, self.ctl_n_layers,
                                      bias=self.bias, batch_first=self.batch_first,
                                      dropout=self.dropout)
        else:
            raise ValueError('Controller type not known. only GRU/LSTM supported')
       
        self.interface = nn.Sequential(nn.Linear(self.hidden_size, self.ctl_output_size),
                                       nn.ReLU())

        self.explicit = ExplicitMem(self.cell_size, self.n_slots, self.n_reads, 
                                    self.hidden_size, self.tau)

        orthogonal(self.interface[0].weight)
        return

    def init_hidden(self, bs, hx=None):
        """Resetting hidden states for each new batch.

        Passing batch_size allows model to be run with variable batch size. (?)
        """
        if hx is None:
            hx = (None, None, None, None)
        (chx, em, em_index, last_read) = hx

        if chx is None:
            h = T.zeros(self.ctl_n_layers, bs, self.hidden_size)   # n_layers * bs * H
            xavier_uniform(h)

            if self.ctl_rnn == 'GRU':
                chx = h
            elif self.ctl_rnn == 'LSTM':
                chx = (h, h)

        if em is None:
            em = self.explicit.init_hidden(bs)

        if em_index is None:
            em_index = np.zeros([bs, self.n_slots])               # bs * N
            em_index.fill(np.nan)

        if last_read is None:
            last_read = T.zeros(bs, self.read_size)  # bs * (RW)
        
        return chx, em, em_index, last_read

    def _forward_once(self, ctl_input, hx, seq_idx):
        """
        # TODO: check dimensions
        ctl_input: bs * 1 * feature_dim
        hx:
            chx: n_layers * bs * H
            em['memory']: bs * W * N
            em_index: bs * N
        seq_idx: int
        """

        (chx, em, em_index) = hx
        _out, chx = self.controller(ctl_input.unsqueeze(1), chx)
        _out = _out.squeeze(1)                                   # bs * H

        _interface = self.interface(_out)                        # bs * (E)

        if self.clip != 0:
            out = T.clamp(_interface, -self.clip, self.clip)
        else:
            out = _interface

        # Interface vector
        k_em = out[:, :self.read_size]                                         # bs * RW
        #  k_bm = out[:, self.read_size:2 * self.read_size]                       # bs * RW
        m_t = out[:, 1 * self.read_size: 1 * self.read_size + self.cell_size]  # bs * W
        #  e_t = out[:, 2 * self.read_size + self.cell_size:]                     # bs * W

        #  print(m_t.shape)
        em_read, em, em_erased, em_index = self.explicit(k_em, em, m_t, chx, 
                                                         em_index, seq_idx)
        #  print(m_t.shape)

        return (out, em_read), (chx, em, em_index) 

    def forward(self, input, hx=(None, None, None, None), save_trace=False):
        is_packed = type(input) is PackedSequence
        if is_packed:
            input, lengths = pad(input, batch_first=self.batch_first)
            max_length = lengths.numpy()[0]  # assuming sorted packed
        else:
            max_length = input.size(1) if self.batch_first else input.size(0)
            lengths = [max_length] * input.size(0) if self.batch_first \
                else [max_length] * input.size(1)

        batch_size = input.size(0) if self.batch_first else input.size(1)

        #  print(batch_size, max_length)
        if not self.batch_first:
            input = input.transpose(0, 1)  # making it batch first

        ctl_hidden, exp_hidden, em_index, l_read = self.init_hidden(batch_size,
                                                                    hx)
 
        outs = [None] * max_length
        read_vecs = [None] * max_length       # flattenned read vector

        all_mem = [None] * max_length     # read weight vecs

        # Recurrent lopp
        for time in range(max_length):
            ctl_input = T.cat([input[:, time, :], l_read], 1)  # bs * feature_dim

            #  import ipdb; ipdb.set_trace()  # BREAKPOINT
            chx, em, m_idx = ctl_hidden, exp_hidden, em_index
            (_out, em_read), (chx, em, m_idx) =\
                self._forward_once(ctl_input, (chx, em, m_idx), time)
            ctl_hidden = chx
            exp_hidden = em
            em_index = m_idx
            
            #  outs[time] = _out

            read_vecs[time] = em_read
            l_read = read_vecs[time].view(batch_size, -1)  # bs * RW
            outs[time] = T.cat([chx.transpose(0, 1).contiguous().view(batch_size, -1),
                                l_read], 1)

            #  print(em_index)

            if save_trace:
                all_mem[time] = (self.explicit.detach(em),
                                 m_idx)
        # ------- end recursion

        outs = T.stack(outs)
        if self.batch_first:
            outs.transpose_(0, 1)

        if is_packed:
            outs = pack(outs, lengths, batch_first=self.batch_first)

        return outs, (ctl_hidden, exp_hidden, em_index, l_read), all_mem

    def __repr__(self):
        """
        TODO: write a more detailed `repr` method
        """
        return super(EmRNN, self).__repr__()
