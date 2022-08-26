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
from lightsaber.data_utils import utils as du
from lightsaber.data_utils import filters as df
from lightsaber import constants as C
import warnings
import os

import logging
log = logging.getLogger()

idx_slice = pd.IndexSlice

from abc import ABC

DEFAULT_FILTER = [df.identity_nd]


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
#         Dataset class and its uitls
# ----------------------------------------------------------------------------
class EmptyDataset(object, ABC):
    def __len__(self):
        return 0


class BaseDataset(object, ABC):
    def __init__(self, 
                 tgt_file, 
                 feat_file, 
                 idx_col, 
                 tgt_col, 
                 feat_columns=None, 
                 time_order_col=None,
                 category_map=C.DEFAULT_MAP,
                 filter=DEFAULT_FILTER,
                 ):
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
        filter: single callable or list/tuple of callables
            how to filter data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: no operation

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
    def _select_features(cls, data, columns, extra_columns=None):
        if columns is not None:
            if is_re(columns):
                _feat_re = columns
            else:
                _feat_re = r"|".join([f"^{x}$" for x in columns])

            try:
                _selected_cols = data.columns[data.columns.str.contains(_feat_re)]
            except Exception:
                warnings.warn("regex mode failed. using raw mode")
                _selected_cols = pd.Series(columns)
            columns = _selected_cols.union(pd.Series(extra_columns))
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
        self.data = self._select_features(self.data, self._feat_columns,
                                          extra_columns=self._time_order_col
                                          )
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

    def __len__(self):
        return len(self.sample_idx)
