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
    """Identity functions

    Args:
        args: variable arguments

    Returns:
        args: same arguments
    """
    return args


def identity_2d(x, y):
    """
    Identity function for 2 variables

    Parameters
    ----------
    x : 
        first param
    y : 
        second param

    Returns
    -------
    x : object
        first param
    y : object
        second param
    """
    return x, y


def identity_3d(x, y, z):
    return x, y, z


def is_re(s:str, 
          strict:bool = False) -> bool:
    """checks if `s` is a valid regex.

    Parameters
    ----------
    s: str
      parameter to check for being a valid regex
    strict: bool
      if strict mode used, anything except regex compile error will throw exception. else functions return False

    Returns
    -------
    bool
        returns True is passed string `s` is a valid reg-ex

    Note
    ----
    ref: https://stackoverflow.com/a/19631067
    """

    try:
        re.compile(s)
        is_valid = True
    except re.error:
        is_valid = False
    except Exception as e:
        if strict:
            raise Exception(e)
        else:
            warnings.warn(f"regex checking incomplete... error while checking: {e}. assuming invalid regex. use strict=True to throw exception")
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
def filter_fillna(data, target, fill_value=0., time_order_col=None):
    """
    Filter function to remove na
    """
    data = data.copy()
    
    idx_cols = data.index.names
    if time_order_col is not None:
        try:
            sort_cols = idx_cols + time_order_col
        except:
            sort_cols = idx_cols + [time_order_col]
    else:
        sort_cols = idx_cols
    
    data.update(data.reset_index()
               .sort_values(sort_cols)
               .groupby(idx_cols[0])
               .ffill())
    
    data.fillna(fill_value, inplace=True)
    return data, target


@functoolz.curry
def filter_flatten(data, target, method='max'):
    log.debug("Starting to flatten")
    # ops = dict(sum='sum', max='max', mean='mean')
    # col_tx = _get_flatten_tx(data, method)
    # data = data.apply(col_tx)
    data = (data.groupby(data.index.names)
                .agg(method))
    #  print(time.time())
    log.debug("Done in flatten")
    return data, target


@functoolz.curry
def filter_flatten_filled_drop_cols(data, target,
                                    aggfunc="sum", 
                                    fill_value=0.0, 
                                    cols_to_drop=C.DEFAULT_DROP_COLS):

    data = data.drop(columns=cols_to_drop, errors='ignore')

    # Fillna
    data, target = filter_fillna(data, target, fill_value=fill_value)

    # Aggfunc
    data, target = filter_flatten(data, target, method=aggfunc)
    return data, target


@functoolz.curry
def filter_preprocessor(data, target, cols=None, preprocessor=None, refit=False):
    if preprocessor is not None:
        all_columns = data.columns
        index = data.index

        # Extracting the columns to fit
        if cols is None:
            cols = all_columns
        _oCols = all_columns.difference(cols)
        xData = data[cols]
    
        # If fit required fitting it
        if refit:
            preprocessor.fit(xData)
            log.info(f'Fitting pre-proc: {preprocessor}')
  
        # Transforming data to be transformed
        try:
            xData = preprocessor.transform(xData)
        except NotFittedError:
            raise Exception(f"{preprocessor} not fitted. pass fitted preprocessor or set refit=True")
        xData = pd.DataFrame(columns=cols, data=xData, index=index)
        
        # Merging other columns if required
        if not _oCols.empty:
            tmp = pd.DataFrame(data=data[_oCols].values, 
                               columns=_oCols,
                               index=index)
            xData = pd.concat((tmp, xData), axis=1)
        
        # Re-ordering the columns to original order
        data = xData[all_columns]
    return data, target


def filt_get_last_index(data, target, 
                        idx_col=['DESY_SORT_KEY', 'INDEX_CLAIM_ORDER'],
                        min_occurence=4
                       ):
    """Filter to get last index claim for each patient.

    Filters are designed to be composible functions such that one can chain filters. 
    Outputs 
    filtered `data` and `target` with entries for only the last index claim

    Parameters
    ----------
    data : DataFrame
        feature data
    target : DataFrame 
        target data
    idx_col: str or int or List of str|int
        index columns
    min_occurence: int
        number of minimum occurence required for an instance to be included

    Returns
    -------
    data: DataFrame
    target: DataFrame
    """
    # last index claim for each patient
    last_claim_idx = (data.reset_index()[idx_col].groupby([idx_col[0]])   # Group by pateint id
                          .max()[idx_col[1]].to_frame()                   # take last index claim order for a patient
                          .reset_index().set_index(idx_col))              # set index to patient id and index_claim_order

    # filter data and keep only last index claim for each patient and its history
    data = data[data.reset_index().set_index(idx_col).index.isin(last_claim_idx.index)]
    
    # remove all patients (last claim index) who have only one claim as it is not useful for med2vec  
    temp = data.reset_index().groupby(idx_col).count().iloc[:,0]
   
    useful_claims_idx = temp[temp>=min_occurence].index 

    data = data[data.index.isin(useful_claims_idx)]
    if target is not None:
        target = target[target.index.isin(data.index)]

    return data, target


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

    def __init__(self, tgt_file, feat_file, 
                 idx_col, tgt_col, 
                 feat_columns=None, time_order_col=None,
                 category_map=C.DEFAULT_MAP,
                 transform=DEFAULT_TRANSFORM, filter=DEFAULT_FILTER,
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
        self._tgt_file = tgt_file
        self._feat_file = feat_file
        
        self._idx_col = idx_col
        self._tgt_col = tgt_col
        self._feat_columns = feat_columns
        self._time_order_col = time_order_col

        self._filter = self._compose(filter, manual=True)
   
        # reading data
        self.read_data()

        # apply filters on datasets
        self.apply_filters()
        
        # Handle categorical columns
        self.data = self.one_hot_encode(self.data, category_map)

        # Book-keeping of number of instances
        self.sample_idx = self.data.index.to_series().drop_duplicates()

        # Runtime configurations 
        self.device = device
        self._transform = self._compose(transform)
        return

    @classmethod    
    def _compose(cls, obj, manual=False):
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

    @classmethod
    def _select_features(cls, data, columns):
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
        if tgt_file is not None:
            self.target = pd.read_csv(tgt_file).set_index(self._idx_col)
            try:
                self.target = self.target[self._tgt_col].to_frame() 
            except AttributeError:
                self.target = self.target[self._tgt_col] 
        else:
            self.target = None
    
        self.data = pd.read_csv(feat_file).set_index(self._idx_col)
        if self.target is not None:
            self.data = self.data.loc[self.target.index, :]   # accounting for the option that target can have lesser number of index than data
        self.data = self._select_features(self.data, self._feat_columns)
        return

    def apply_filters(self):
        for f in reversed(self._filter):
            self.data, self.target = f(self.data, self.target)
        return

    def one_hot_encode(self, data, category_map):
        _one_hot_cols = []
        for col, categories in iteritems(category_map):
            if col in data.columns:
                data.loc[:, col] = pd.Categorical(data[col], categories=categories)
                _one_hot_cols.append(col)
        data = pd.get_dummies(data, columns=_one_hot_cols)
        return data 

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
            target = pd.DataFrame([np.nan]*len(data.index), index=data.index)

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


