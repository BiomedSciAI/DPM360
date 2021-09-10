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

import os
import sys
import numpy as np
import pandas as pd
import pickle
from toolz import functoolz

from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

from lightsaber import constants as C
from lightsaber.data_utils import pt_dataset as ptd 
from lightsaber.data_utils.pt_dataset import (filter_flatten_filled_drop_cols,
                                              filter_preprocessor)

import warnings
import logging
log = logging.getLogger()

DEFAULT_DEVICE = 'cpu'
DEFAULT_FILTER = None
DEFAULT_TRANSFORM = [functoolz.identity]

class SKDataLoader(object):
    """Custom data loaders for scikit-learn"""

    def __init__(self, 
                 tgt_file, feat_file, idx_col, tgt_col, 
                 feat_columns=None, 
                 time_order_col=None, 
                 category_map=C.DEFAULT_MAP, 
                 filter=DEFAULT_FILTER,
                 fill_value=0.,
                 flatten=C.DEFAULT_FLATTEN, 
                 cols_to_drop=C.DEFAULT_DROP_COLS,
                 ):
        """
        Parameters
        ----------
        tgt_file:
            target file path
        feat_file:
            feature file path
        idx_col:
            columns to specify the unique examples from the feature and target set
        tgt_col:
            columns to specify the target column from the target set.
        feat_columns:
            feature columns to select from. either list of columns (partials columns using `*` allowed) or a single regex
            Default: `None` -> implies all columns
        time_order_col:
            column(s) that signify the time ordering for a single example.
            Default: `None` -> implies no columns
        category_map:
            dictionary of column maps
        filter: single callable or list/tuple of callables
            how to filter data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: no operation
        fill_value:
            pandas compatible function or value to fill missing data
        flatten:
            Functions to aggregate and flatten temporal data
        cols_to_drop:
            list of columns to drop
        """

        self._tgt_file = tgt_file
        self._feat_file = feat_file
        self._idx_col = idx_col
        self._tgt_col = tgt_col
        self._feat_columns = feat_columns
        self._time_order_col= time_order_col
        self._category_map = category_map
        
        # Enforing a flatten function to make sure sklearn modules gets a
        # flattended data
        _filter_flatten_filled_drop_cols = filter_flatten_filled_drop_cols(cols_to_drop=cols_to_drop,
                                                                                aggfunc=flatten,
                                                                                fill_value=fill_value)
        self._filter = []
        if filter is not None:
            if isinstance(filter, (list, tuple)):
                self._filter += filter
            else:
                self._filter.append(filter)
        self._filter.append(_filter_flatten_filled_drop_cols)

        # Reading data
        self.read_data()
        return

    def read_data(self):
        device = DEFAULT_DEVICE
        transform = DEFAULT_TRANSFORM

        self._dataset = ptd.BaseDataset(self._tgt_file, 
                                  self._feat_file, 
                                  self._idx_col, 
                                  self._tgt_col,
                                  feat_columns=self._feat_columns, 
                                  time_order_col=self._time_order_col,
                                  category_map=self._category_map,
                                  filter=self._filter,
                                  transform=transform,
                                  device=device
                                 )
        return

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def sample_idx(self):
        return self._dataset.sample_idx

    def __len__(self):
        return len(self._dataset)

    def get_data(self):
        X = self._dataset.data
        y = self._dataset.target
        return X, y

    def get_patient(self, patient_id):
        p_idx = self._dataset.sample_idx.index.get_loc(patient_id)
        full_X, full_y = self.get_data()
        p_X = full_X.iloc[[p_idx]]
        if full_y is not None:
            p_y = full_y.iloc[[p_idx]]
        else:
            p_y = None
        return p_X, p_y
