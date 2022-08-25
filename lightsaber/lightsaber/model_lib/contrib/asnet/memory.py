#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as T
import torch.nn.functional as F

from torch import nn
from .utils import batch_cosine, softmax_nd, gumbel_softmax_nd

import numpy as np


class MixGate(nn.Module):
    def __init__(self, vec_size, bias=True):
        """
        Custom Gate between two vectors to generate a weighted vector

        TODO: check code
        """
        nn.Module.__init__(self)
        self.vec_size = vec_size
        self.bias = bias

        self.build_model()
        return

    def build_model(self):
        self.g_linear = nn.Linear(2 * vec_size, 1, bias=self.bias)
        self.g_act = nn.Sigmoid()

        self.o_linear = nn.Linear(2 * vec_size, vec_size, bias=self.bias)
        self.o_act = nn.ReLU()
        pass

    def forward(self, vec1, vec2):
        _vec_shape = vec1.shape
        dimBy = len(_vec_shape) - 1

        g_out = self.g_act(self.g_linear(T.cat([vec1, vec2], dimBy))) 
     
        g_vec1 = g_out * vec1
        g_vec2 = (1 - g_out) * vec2

        o_out = self.o_act(self.o_linear(T.cat([g_vec1, g_vec2], dimBy)))
        return o_out, g_out

    def get_regularization_weights(self):
        """
        Regularization weights on vec2
        """
        return T.cat([self.g_linear.weight[:, self.vec_size:],
                      self.o_linear.weight[:, self.vec_size:]], 1) 


class BaseMemory(nn.Module):

    def __init__(self, n_dims, n_slots, n_reads, tau=1.):
        """Base Memory class to be abstracted by other memory types"""
        nn.Module.__init__(self)

        self.n_dims = n_dims
        self.n_slots = n_slots
        self.n_reads = n_reads
        self.tau = tau

        #  self.EE = 1 - T.eye(self.n_slots).unsqueeze(0)  # (1 * n * n)

    def content_weightings(self, k, M, strengths=None, measure='softmax', 
                           reverse=False):
        """
        Paramters
        ---------
        k:   key(s) to compare against (bs * r * W)
        M: memory content (bs * W * N)
        strengths: additional strengths for each location
        measure: gumbel-softmax/softmax
        inverse: if True, calculate dissimilarity
        """
        dist = batch_cosine(M.transpose(1, 2), k)   # bs * R * N

        if strengths:
            dist.add_(strengths.unsqueeze(2))

        if reverse:
            dist.mul_(-1)

        if measure == 'softmax':
            ret = softmax_nd(dist, 2)                        # bs * r * N
        elif measure == 'gumbel-softmax':
            ret = gumbel_softmax_nd(dist, 2, tau=self.tau)   # bs * r * N
        return ret

    def init_hidden(self, bs=1, hx=None):
        """
        Initializaing Memory.

        Parameters
        ----------
        bs: batch size
        hx: hidden states for memory. 
            if not provided, it generates a zero matrix. else, use the values in hx.
        """
        if hx is None:
            mem = {'memory': T.rand(bs, self.n_dims, self.n_slots),          # bs * N * W
                   'usage': T.zeros(bs, self.n_slots, requires_grad=False),   # bs * N
                   'read_weights': T.zeros(bs, self.n_reads, self.n_slots),   # bs * R * N
                   }
        else:
            mem = dict()
            for k in ['memory', 'usage', 'read_weights', 'write_weights']:
                mem[k] = hx[k].clone()
        return mem

    def read(self, k, mem, measure):
        read_wt = self.content_weightings(k, mem['memory'], measure=measure)

        mem['read_weights'] = read_wt         

        read_vec = T.bmm(mem['read_weights'],                 # bs * R * N
                         mem['memory'].transpose(1, 2)        # bs * W * N
                         )                                    # bs * R * W
        return read_vec, mem

    @staticmethod
    def detach(mem):
        return {k: mem[k].detach() for k in mem}

    def write(self, x):
        raise NotImplementedError()

    @staticmethod
    def update_mem(M, wt, mt, et):
        """
        Returns updated memory based on 
           ::math::`M \circle (E - et^T wt) + mt^t wt`

        Parameters
        ----------
        M: bs * W * N
        wt: bs * 1 * N
        mt: bs * 1 * W
        et: bs * 1 * W
        """
        weighted_resets = T.bmm(et.transpose(1, 2), wt)           # bs * W * N
        reset_gate = (1 - weighted_resets)   # diffused read over dimensions   
        
        M = M * reset_gate                                        # bs * W * N
        M = M + T.bmm(mt.transpose(1, 2), wt)                     # bs * W * N
        return M


class ExplicitMem(BaseMemory):

    def __init__(self, n_dims, n_slots, n_reads, hidden_size, tau, alpha_usage=0.7):
        """TODO: to be defined1. """
        BaseMemory.__init__(self, n_dims, n_slots, n_reads, tau)
        self.hidden_size = hidden_size
        self.alpha_usage = alpha_usage

        # TODO: add bias as parameters
        self.usage_gate = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                        nn.Sigmoid())

    def _get_usage(self, ut, read_w):
        """
        ut: bs * N
        read_w: bs * r * N
        """
        r_w = read_w.detach()
        r_w = r_w.sum(dim=1)      # bs * N
        ut = self.alpha_usage * ut + (1 - self.alpha_usage) * r_w
        return ut

    def read(self, k, mem):
        return super(ExplicitMem, self).read(k, mem, measure='gumbel-softmax')

    def write(self, m_t, mem, hx, m_idx, seq_idx, bs):
        """
        m_t: bs * 1 * w
        mem['memory']: bs * W * N
        hx: bs * H
        """
        len_filled = len(m_idx[~np.isnan(m_idx)])
        
        # Usage based update
        u_t = self._get_usage(mem['usage'], mem['read_weights'])        # bs * N
        if len_filled < self.n_slots:
            # No grad loop
            # memory not filled 
            write_idx = T.ones(bs, 1).type(T.LongTensor) * len_filled
            write_wt = T.FloatTensor(bs, self.n_slots)
            #  import ipdb; ipdb.set_trace()  # BREAKPOINT
            write_wt.zero_()
            write_wt.scatter_(1, write_idx, 1)        # nograd

            m_erased = None
            mem['memory'] = T.where(write_wt[:, None, :] == 1,
                                    m_t.transpose(1, 2), mem['memory'])
            #  mem['memory'][:, :, write_idx] = m_t.squeeze(1)  # inplace operation
        else:
            # FIXME
            soft_wt = self.content_weightings(m_t, mem['memory'], 
                                              reverse=True)                 # bs * 1 * N

            gamma = self.usage_gate(hx[-1, :, :])                                     # bs * 1
            erase_vec = T.ones(bs, 1, self.n_dims, requires_grad=False)     # bs * 1 * W

            # retention probability
            soft_wt = soft_wt + (gamma * u_t).unsqueeze_(1)                 # bs * 1 * N  
            # removal probability
            write_wt = gumbel_softmax_nd(1 - soft_wt, 2, tau=self.tau)      # bs * 1 * N  

            m_erased = T.bmm(write_wt, mem['memory'].transpose(1, 2))       # bs * 1 * W 

            Mt = self.update_mem(mem['memory'], write_wt, m_t, erase_vec)   # bs * W * N
            mem['memory'] = Mt

            #  with T.no_grad():
        _det_write_wt = write_wt.detach().squeeze(1)                    # bs * N
        m_idx[_det_write_wt.numpy() == 1] = seq_idx
            
        mem['usage'] = u_t * (1 - _det_write_wt)  # resetting the usage
        # ------------ end no grad operation
        # ---------------- end mem loop
        return mem, m_idx, m_erased

    def forward(self, k_r, mem, m_t, hx, m_idx, seq_idx):
        """
        Paramters
        ---------
        k_r: read key(s) to compare against mem (bs * RW)
        mem: current memory content             (`memory`: bs * W * N)
        m_t: candidate memory                   (bs * W)
        m_idx: current indices of memories      (bs * N)
        seq_idx: current sequence index 
        """
        bs = k_r.size()[0]  # batch size
        read_keys = F.tanh(k_r.contiguous().view(bs, self.n_reads, self.n_dims))  # transformed key
        write_m = F.tanh(m_t.contiguous().view(bs, 1, self.n_dims))  # write vector

        # writing the vectors
        mem, m_idx, m_erased = self.write(write_m, mem, hx, m_idx, 
                                          seq_idx, bs)

        # reading the vector
        m_read, mem = self.read(read_keys, mem)

        return m_read, mem, m_erased, m_idx


class BlurredMem(BaseMemory):

    def __init__(self, n_dims, n_slots, n_reads):
        """TODO: to be defined1. """
        BaseMemory.__init__(self, n_dims, n_slots, n_reads)

        self.write_gate = MixGate(n_dims)  # gate to generate new candidate memory

    def read(self, k, mem):
        return super(BlurredMem, self).read(k, mem, measure='softmax')

    def write(self, m_t, mem, m_erased, erase_vec, bs):
        """
        m_t: bs * 1 * W
        mem['memory']: bs * W * N
        m_erased: bs * 1 * W
        erase_vec: bs * 1 * W
        """

        if m_erased is None:
            write_m = m_t
        else:
            write_m, _ = self.write_gate(m_t, m_erased)

        write_wt = self.content_weightings(write_m, mem['memory'], reverse=True)  # bs * 1 * N
        Mt = self.update_mem(mem['memory'], write_wt, m_t, erase_vec)  # bs * W * N
        mem['memory'] = Mt
        return mem

    def forward(self, k_r, mem, m_t, m_erased, e_t):
        """
        Paramters
        ---------
        k_r: key(s) to compare against mem (r * W)
        mem: current memory content
        m_t: candidate memory
        m_erased: erased memory from previous bank
        e_t: erase vector
        """
        bs = k_r.size()[0]  # batch size
        read_keys = F.tanh(k_r.contiguous().view(bs, self.n_reads, self.n_dims))  # transformed key
        write_m = F.tanh(m_t.contiguous().view(bs, 1, self.n_dims))  # write vector
        erase_vec = softmax_nd(e_t.contiguous().view(bs, 1, self.n_dims), 2)  # bs * 1 * W
        
        # writing the vectors
        mem = self.write(write_m, mem, m_erased, erase_vec, bs)

        # reading the vector
        m_read, mem = self.read(read_keys, mem)

        return m_read, mem       
