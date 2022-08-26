#!/usr/bin/env python2
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
from __future__ import absolute_import

import re
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from six import iteritems
from toolz import functoolz

from sklearn.exceptions import NotFittedError
from torch import is_tensor
import torch as T
from lightsaber.data_utils import utils as du
from lightsaber.data_utils import filters as df
from lightsaber.data_utils.dataset.base import EmptyDataset, BaseDataset
from lightsaber import constants as C
import warnings
import os

import logging
log = logging.getLogger()

idx_slice = pd.IndexSlice


# -----------------------------------------------------------------------------
#         List of pre-defined transforms
# ----------------------------------------------------------------------------
@functoolz.curry
def transform_default(data, time_order_col, fill_value=0.):
    raise DeprecationWarning("deprecated. this will be dropped in v0.3. use [transform_drop_cols, transform_fill]")
    data = (data.drop(columns=time_order_col)   # REMOVING Time order col
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
    col_tx = df._get_flatten_tx(data, method)
    data = data.apply(col_tx)
    return data


DEFAULT_TRANSFORM = [transform_drop_cols, transform_fillna]
DEFAULT_FILTER = [df.identity_nd]


# -----------------------------------------------------------------------------
#         Dataset class and its uitls
# ----------------------------------------------------------------------------
class PTEmptyDataset(EmptyDataset, Dataset):
    pass
    

class PTBaseDataset(EmptyDataset, Dataset):
    def __init__(self, 
                 tgt_file, 
                 feat_file, 
                 idx_col, 
                 tgt_col, 
                 feat_columns=None, 
                 time_order_col=None,
                 category_map=C.DEFAULT_MAP,
                 transform=DEFAULT_TRANSFORM, 
                 filter=DEFAULT_FILTER,
                 device='cpu'):
        """Base dataset class

        Parameters
        ----------
        tgt_file:
            target file path
        feat_file:
            feature file path
        idx_col: str or List[str]
            index columns in the data. present in both `tgt_file` and `feat_file`
        tgt_col: str or List[str]
            target column present in `tgt_file` 
        feat_columns:
            feature columns to select from. either a single regex or list of columns (partial regex that matches the complete column name is ok. e.g. `CCS` would only match `CCS` whereas `CCS.*` will match `CCS_XYZ` and `CCS`) 
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

        Examples
        --------
        
        Example of feature columns.
        >>> df = pd.DataFrame(columns = ['CCS_128', 'CCS', 'AGE', 'GENDER'])
        >>> feat_columns = ['CCS_.*', 'AGE'] 
        >>> BaseDataset._select_features(df, feat_columns)
        ['CCS_128', 'AGE']

        >>> feat_columns = ['CCS', 'AGE'] # would select 'CCS' and 'AGE'
        >>> BaseDataset._select_features(df, feat_columns)
        ['CCS', 'AGE']
        """
        super().__init__(tgt_file=tgt_file, 
                         feat_file=feat_file, 
                         idx_col=idx_col, 
                         tgt_col=tgt_col, 
                         feat_columns=feat_columns, 
                         time_order_col=time_order_col,
                         category_map=category_map,
                         transform=transform, 
                         filter=filter,
                         device=device
                         )

        # Runtime configurations 
        self.device = device
        self._transform = self._compose(transform)
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

        data = self.data.loc[[idx], :]
        if self.target is not None:
            target = self.target.loc[[idx], :].astype('float')
        else:
            target = pd.DataFrame([np.nan] * len(data.index), index=data.index)

        # print(data.head())
        # import ipdb; ipdb.set_trace()
        
        if self._time_order_col is not None:
            _sort = False
            if isinstance(self._time_order_col, list):
                _sort = data.columns.isin(self._time_order_col).sum() == len(self._time_order_col)
            else:
                _sort = self._time_order_col in data.columns
            
            if _sort:
                data = data.sort_values(self._time_order_col)

        data = self._transform(data).astype('float')

        data_t = T.FloatTensor(np.atleast_2d(data.values)).to(device)
        target_t = T.LongTensor(target.values).squeeze().to(device)
        length = data_t.size(0)
        return data_t, target_t, length, idx

    def get_patient(self, patient_id):
        p_idx = self.sample_idx.index.get_loc(patient_id)
        p_x, p_y, p_lengths, _ = self.__getitem__(p_idx)
        p_x.unsqueeze_(0)  # adding dummy dimension for batch
        p_y.unsqueeze_(0)  # adding dimension for batch
        p_lengths = [p_lengths, ]
        return p_x, p_y, p_lengths, p_idx


class PTAutoregressionDataset(Dataset):
    """Simple Dataset class.
    """
    #  _allowed_modes = ['all', 'individual']
    _allowed_modes = ['all']

    def __init__(self,
                 fname,
                 id_cols=C.ID_COLS,
                 data_cols=None,
                 cont_cols=None,
                 unmodeled_cols=None,
                 tgt_cols=C.TGT_COLS,
                 data_transform=None,
                 tgt_transform=None,
                 mode=C.DATA_MODE,
                 device='cpu',
                 ):
        """Simple Dataset

        Parameters
        ----------
        fname: str/Path object
            filename
        id_cols: list. Default: ['id', 'time']
            cols to be used as index. only first index used to identify unique patients
        data_col: list / None.  Default: None
            if provided, only  those columns used. None (Default) implies all columns except target and index columns
        tgt_cols: list. Default: ['X']
            column(s) to be used as the target
        device: pytorch compatible device id. Default: 'cpu'
        """
        self.device = device

        self.id_cols = id_cols
        self.tgt_cols = tgt_cols
        self.tgt_transform = tgt_transform
        self.data_transform = data_transform if data_transform is not None else tgt_transform # backward compatibility
        self.mode = mode
        assert self.mode in self._allowed_modes, f"data mode needs to be from {self._allowed_modes}"

        self._data = pd.read_csv(fname).set_index(self.id_cols)
        _avlbl_cols = self._data.columns.difference(set(self.tgt_cols)
                                                    .union(self.id_cols))
        if data_cols is None:
            self.data_cols = _avlbl_cols
        else:
            self.data_cols = data_cols
            assert pd.Series(self.data_cols).isin(_avlbl_cols).all(), f"Available cols: {_avlbl_cols}"
        
        if cont_cols is None:
            self.cont_cols = [self.data_cols[0],]
        else:
            self.cont_cols = cont_cols
            
        if unmodeled_cols is not None:
            import ipdb; ipdb.set_trace()  # BREAKPOINT
            assert (#(not pd.Series(unmodeled_cols).isin(self.data_cols).any()) &
                    (pd.Series(unmodeled_cols).isin(_avlbl_cols).all())), f"Available cols: {_avlbl_cols}"
        self.unmodeled_cols = unmodeled_cols

    @property
    def ids(self):
        if not hasattr(self, '_ids'):
            if self.mode == 'individual':
                self._ids = self._data.index.values
            elif self.mode == 'all':
                self._ids = self._data.index.levels[0].values
        return self._ids

    def __len__(self):
        return len(self.ids)
    
    def get_unmodeled(self, i):
        _ids = self.ids
        idx = _ids[i] # Finding the id
        
        if self.mode == 'individual':
            unmodeled = self._data.loc[idx, self.unmodeled_cols].values
        else:
            unmodeled = self._data.loc[pd.IndexSlice[idx, :], self.unmodeled_cols].values
        unmodeled_t = T.FloatTensor(np.atleast_2d(unmodeled)).to(self.device)
        return unmodeled_t

    def __getitem__(self, i):
        _ids = self.ids
        idx = _ids[i] # Finding the id

        if self.mode == 'individual':
            row = self._data.loc[idx, :].to_frame().T
        else:
            row = self._data.loc[pd.IndexSlice[idx, :], :]

        data = row.loc[:, self.data_cols]
        target = row.loc[:, self.tgt_cols]
        if self.tgt_transform is not None:
            # backward compatibility
            tmp = data.copy()
            tmp[self.cont_cols] = self.data_transform(tmp[self.cont_cols].values.reshape(-1, len(self.cont_cols)))
            # data = np.hstack((self.tgt_transform(data[self.cont_cols].values.reshape(-1, len(self.cont_cols))),
            #                   data[:, 1:]))
            data = tmp.values
            if isinstance(self.tgt_transform, dict):
                target = np.hstack([self.tgt_transform[k](target[np.atleast_1d(self.tgt_cols[k])])
                                    for k in self.tgt_transform])
            else:
                target = self.tgt_transform(target.values)
            del(tmp)
        else:
            data = data.values
            target = target.values

        data_t = T.FloatTensor(np.atleast_2d(data)).to(self.device)
        target_t = T.FloatTensor(np.atleast_1d(target)).to(self.device)
        length = data_t.size(0)
        
        ret = [data_t, target_t, length, idx]
        
        # FIXME: may need this later
        # if self.unmodeled_cols is not None:
        #     unmodeled_t = T.FloatTensor(np.atleast_2d(row.loc[:, self.unmodeled_cols].values)).to(self.device)
        #     ret.append(unmodeled_t)
        return tuple(ret)

# -----------------------------------------------------------------------------
#        Some collate functions
# ----------------------------------------------------------------------------
def collate_fn(batch): 
    """Provides mechanism to collate the batch

    ref: https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
    
    Puts data, and lengths into a packed_padded_sequence then returns
    the packed_padded_sequence and the labels.

    Parameters
    ----------
    batch : List[Tuples]
        [(*data, target)] data: all the differnt data input from `__getattr__`  target: target y

    Returns
    -------
    Tuple
        (dx, dy, lengths, idx)
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


