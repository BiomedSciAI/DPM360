#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from six import iteritems
from toolz import functoolz

from torch import is_tensor
import torch as T
from lightsaber.data_utils import utils as du
from lightsaber import constants as C
import warnings
import os

import logging
log = logging.getLogger()

idx_slice = pd.IndexSlice


# -----------------------------------------------------------------------------
#         Some basic utils
# ----------------------------------------------------------------------------
def identity_nd(*args):
    return args


def identity_2d(x, y):
    return x, y


def identity_3d(x, y, z):
    return x, y, z


def is_re(s):
    """
    checks if `s` is a valid regex.

    ref: https://stackoverflow.com/a/19631067
    """

    try:
        re.compile('[')
        is_valid = True
    except re.error:
        is_valid = False
    return is_valid


# -----------------------------------------------------------------------------
#         List of pre-defined filters
# ----------------------------------------------------------------------------
def _get_flatten_tx(data, method):
    '''
    Assigns a flatten function to each col in the data dataframe
    TODO: MG I think this function needs some work
    '''
    if 'sum' in method:
        cols = {col: 'sum' for col in data 
                if (col.startswith('ICD') or 
                    col.startswith('PROC') or 
                    col.startswith('LENGTH'))}
        for i in data.columns:
            if i not in cols:
                if i.startswith('EMB'):
                    cols[i] = 'mean'
                else:
                    cols[i] = 'max'

    if 'max' in method:
        cols = dict()
        for i in data.columns:
            if i.startswith('EMB'):
                cols[i] = 'mean'
            else:
                cols[i] = 'max'

    if 'mean' in method:
        cols = {col: 'mean' for col in data 
                if (col.startswith('ICD') or 
                    col.startswith('PROC') or 
                    col.startswith('LENGTH') or 
                    col.startswith('EMB'))}
        for i in data.columns:
            if i not in cols: 
                cols[i] = 'max'
    return cols


@functoolz.curry
def filter_flatten_filled_drop_cols(data, target,
                                    aggfunc="sum", 
                                    fill_value=0.0, 
                                    cols_to_drop=C.DEFAULT_DROP_COLS):
    log.debug("Starting to flatten")

    data = data.drop(columns=cols_to_drop, errors='ignore')

    #  print(time.time())
    data = (data.groupby(data.index.names)
                .fillna(fill_value))
    data = (data.groupby(data.index.names)
                .agg(aggfunc))
    #  print(time.time())
    log.debug("Done in flatten")
    return data, target


def filt_get_last_index(data, target):
    """
    Filter to get last index claim for each patient.

    Filters are designed to be composible functions such that one can chain filters

    Parameters
    ----------
    data : feature data
    target : target data

    Returns
    -------
    filtered `data` and `target` with entries for only the last index claim
    """
    idx = ['DESY_SORT_KEY', 'INDEX_CLAIM_ORDER']

    # last index claim for each patient
    last_claim_idx = (data.reset_index()[idx].groupby([idx[0]])   # Group by pateint id
                          .max()[idx[1]].to_frame()          # take last index claim order for a patient
                          .reset_index().set_index(idx))     # set index to patient id and index_claim_order

    # filter data and keep only last index claim for each patient and its history
    data = data[data.reset_index().set_index(idx).index.isin(last_claim_idx.index)]
    
    # remove all patients (last claim index) who have only one claim as it is not useful for med2vec  
    temp = data.reset_index().groupby(du.IDX_COLS).count().iloc[:,0]
    
    useful_claims_idx = temp[temp>=4].index 
    data = data[data.index.isin(useful_claims_idx)]

    target = target[target.index.isin(data.index)]

    return data, target

# -----------------------------------------------------------------------------
#         List of pre-defined transforms
# ----------------------------------------------------------------------------
@functoolz.curry
def transform_default(data, fill_value=0.):
    raise DeprecationWarning("deprecated. use [transform_drop_cols, transform_fill]")
    data = (data.drop(columns=du.TIME_ORDER_COL)   # REMOVING Time order col
                .fillna(method='ffill')    # fllling up NAN
                .fillna(method='bfill')
                .fillna(fill_value)
           )
    return data


@functoolz.curry
def transform_drop_cols(data, cols_to_drop=C.DEFAULT_DROP_COLS):
    data = data.drop(columns=cols_to_drop, errors='ignore')
    return data


@functoolz.curry
def transform_fillna(data, fill_value=0.):
    data = (data.fillna(method='ffill')    # fllling up NAN
                .fillna(method='bfill')
                .fillna(fill_value)
           )
    return data


@functoolz.curry
def transform_flatten(data, method='max'):
    """Transform data to flatten data by last value

    Parameters
    ----------
    data : feature values

    Returns
    -------
    flattend data

    """
    # ops = dict(sum='sum', max='max', mean='mean')
    col_tx = _get_flatten_tx(data, method)
    data = data.apply(col_tx)
    return data


DEFAULT_TRANSFORM = [transform_drop_cols, transform_fillna]
DEFAULT_FILTER = [identity_nd]


# -----------------------------------------------------------------------------
#         Dataset class and its uitls
# ----------------------------------------------------------------------------
class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    

class BaseDataset(Dataset):
    """Base dataset"""

    def __init__(self, tgt_file, feat_file, 
                 idx_col, tgt_col, 
                 feat_columns=None, time_order_col=None,
                 category_map=C.DEFAULT_MAP,
                 transform=DEFAULT_TRANSFORM, filter=DEFAULT_FILTER,
                 device='cpu'):
        """Base Dataset

        Parameters
        ----------
        tgt_file:
            target file path
        feat_file:
            feature file path
        feat_columns:
            feature columns to select from. either list of columns (partials columns using `*` allowed) or a single regex
            Default: `None` -> implies all columns
        time_order_col:
            column(s) that signify the time ordering for a single example.
            Default: `None` -> implies no columns 
        category_map:
            dictionary of column maps
        transform: single callable or list/tuple of callables
            how to transform data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: drop `lightsaber.constants::DEFAULT_DROP_COLS` and fillna
        filter: single callable or list/tuple of callables
            how to filter data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: no operation
        device: str
            valid pytorch device. `cpu` or `gpu`
        """
        self._tgt_file = tgt_file
        self._feat_file = feat_file
        
        self._idx_col = idx_col
        self._tgt_col = tgt_col
        self._feat_columns = feat_columns
        self._time_order_col = time_order_col

        self._transform = self._compose(transform)
        self._filter = self._compose(filter, manual=True)

        self.device = device
   
        # reading data and applying filters
        self.read_data()

        # apply filters on datasets
        self.apply_filters()
        
        # transform categorical columns
        self.one_hot_encode(category_map)

        self.sample_idx = self.target.index.to_series()
        return
    
    def _compose(self, obj, manual=False):
        if obj is None:
            obj = identity_nd
        if isinstance(obj, (list, tuple)):
            if manual:
                pass
            else: 
                obj = functoolz.compose(*obj)
        else:
            if manual:
                obj = [obj]
            else:
                pass
        return obj

    def _select_features(self, data, columns):
        if columns is not None:
            if is_re(columns):
                _feat_re = columns
            else:
                _feat_re = r"|".join([f"^{x}$" for x in columns])

            try:
                _selected_cols = data.columns[data.columns.str.contains(_feat_re)]
            except Exception:
                warnings.warn("regex mode failed. using raw mode")
                _selected_cols = columns
            columns = _selected_cols
            # Assume that feature columns is an inclusive list
            return data[columns]
        else:
            return data

    def read_data(self):
        tgt_file, feat_file = self._tgt_file, self._feat_file
        self.target = pd.read_csv(tgt_file).set_index(self._idx_col)
        try:
            self.target = self.target[self._tgt_col].to_frame() 
        except AttributeError:
            self.target = self.target[self._tgt_col] 
    
        self.data = pd.read_csv(feat_file).set_index(self._idx_col)
        self.data = self.data.loc[self.target.index, :]   # accounting for the option that target can have lesser number of index than data
        self.data = self._select_features(self.data, self._feat_columns)
        return

    def apply_filters(self):
        for f in reversed(self._filter):
            self.data, self.target = f(self.data, self.target)
        return

    def one_hot_encode(self, category_map):
        _one_hot_cols = []
        for col, categories in iteritems(category_map):
            if col in self.data.columns:
                self.data.loc[:, col] = pd.Categorical(self.data[col], categories=categories)
                _one_hot_cols.append(col)
        self.data = pd.get_dummies(self.data, columns=_one_hot_cols)
        return

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            data_t, target_t, _, _ = self.__getitem__(0)
            feat_dim = data_t.shape[-1]
            try:
                target_dim = target_t.shape[-1]
            except Exception:
                target_dim = 1
            self._shape = (feat_dim, target_dim)
        return self._shape

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, i):
        device = self.device
        # if is_tensor(idx):
        #     idx = idx.tolist()
        idx = self.sample_idx.iloc[i]

        target = self.target.loc[idx, :]
        data = self.data.loc[[idx], :]
        # print(data.head())
        # import ipdb; ipdb.set_trace()
        
        if self._time_order_col is not None:
            _sort = False
            if isinstance(self._time_order_col, list):
                _sort =  data.columns.isin(self._time_order_col).sum() == len(self._time_order_col)
            else:
                _sort = self._time_order_col in data.columns
            
            if _sort:
                data = data.sort_values(self._time_order_col)

        data = self._transform(data)

        data_t = T.FloatTensor(np.atleast_2d(data.values)).to(device)
        target_t = T.LongTensor(target.values).squeeze().to(device)
        length = data_t.size(0)
        return data_t, target_t, length, idx


# -----------------------------------------------------------------------------
#        Some collate functions
# ----------------------------------------------------------------------------
def collate_fn(batch):
    """
    Provides mechanism to collate the batch

    ref: https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
    Puts data, and lengths into a packed_padded_sequence then returns
    the packed_padded_sequence and the labels.

    Parameters
    ----------
    batch: (list of tuples) [(*data, target)].
         data: all the differnt data input from `__getattr__`
         target: target y

    Returns
    -------
    packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
    target: (Tensor).

    """
    pad = C.PAD

    if len(batch) == 1:
        dx_t, dy_t, lengths, idx = batch[0]
        #  sigs = sigs.t()
        dx_t.unsqueeze_(0)
        dy_t.unsqueeze_(0)
        lengths = [lengths]
        idx = np.atleast_2d(idx)

    else:
        dx_t, dy_t, lengths, idx = zip(*[(dx, dy, length, idx)
                                         for (dx, dy, length, idx) in sorted(batch, key=lambda x: x[2],
                                                                             reverse=True)])
        max_len, n_feats = dx_t[0].size()
        device = dx_t[0].device
        
        dx_t = [T.cat((s, T.empty(max_len - s.size(0), n_feats, device=device).fill_(pad)), 0)
                if s.size(0) != max_len else s
                for s in dx_t]
        dx_t = T.stack(dx_t, 0).to(device)  # bs * max_seq_len * n_feat

        dy_t = T.stack(dy_t, 0).to(device)  # bs * n_out

        # Handling the other variables
        lengths = list(lengths)
        idx = np.vstack(idx) # bs * 1
    return dx_t, dy_t, lengths, idx
