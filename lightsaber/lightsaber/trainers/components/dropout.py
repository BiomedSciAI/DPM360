#!/usr/bin/env python3

from typing import Optional

import math
import numpy as np
import pandas as pd
from typing import tuple, list, optional
import itertools
import warnings

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import packedsequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


class BaseLockedDropout(jit.ScriptModule):
    """A Mixing class to enable dropouts that can be locked by setting dropout masks"""
    #  __constants__ = ["dropout", "batch_first", "variational"]
    
    def __init__(self,
                 dropout: float,
                 batch_first: Optional[bool]=True,
                 variational: Optional[bool]=True):
        super().__init__()
        self.dropout = dropout
        self._batch_first = batch_first
        self._variational = variational

        # other variables for future use
        self._is_mask_set = False
        self.mask = torch.FloatTensor()
        self.mask.requires_grad_(False)

    def reset_mask(self):
        self._is_mask_set = False
        self.mask = torch.FloatTensor()
        return

    def set_mask(self, mask):
        self._is_mask_set = True
        self.mask = mask

    @jit.script_method
    def _apply_mask(self, x:torch.Tensor) -> torch.Tensor:
        if not self.training and (self.dropout <= 0. or not self._variational):
            return x

        mask = self._get_mask(x)
        x = (x * mask) / (1 - self.dropout)
        return x
        
    def _get_mask(self, x):
        raise NotImplementedError("Implement this in your subclass")

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.dropout})"
        return s


class LockedDropout(BaseLockedDropout):

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_mask(x)
        return x
            
    @jit.script_method
    def _get_mask(self, x:torch.Tensor) -> torch.Tensor:
        if not self._is_mask_set:
            mask = (x.data.new_empty(x.size())
                    #  .new_empty(*x.shape)
                    .bernoulli_(1 - self.dropout))
            #  self.mask.data = mask
            self.set_mask(mask)
            # self.mask = mask
            # self._is_mask_set = True
        return self.mask


class VariationalLockedDropout(BaseLockedDropout):

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_mask(x)
        return x
            
    @jit.script_method
    def _get_mask(self, x):
        if len(x.shape) == 3:
            if self._batch_first:
                batch_dim = 0
                locked_dim = 1
            else:
                batch_dim = 1
                locked_dim = 0
        else:
            raise NotImplementedError("Only implemented for 3D tensors. For more general purpose use ::mod::LockedDropout")

        if not self._is_mask_set:
            mask = (x.data
                     .new_empty(x.size(batch_dim), x.size(-1))
                     .bernoulli_(1 - self.dropout))
            #  self.mask.data = mask
            self.set_mask(mask)
            # self.mask = mask
            # self._is_mask_set = True
        mask = self.mask.unsqueeze(locked_dim).expand_as(x)
        return mask




