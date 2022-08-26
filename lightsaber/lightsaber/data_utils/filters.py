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
from lightsaber import constants as C
import warnings
import os

import logging
log = logging.getLogger()

idx_slice = pd.IndexSlice


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
        except Exception:
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
