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
from lightsaber.data_utils.pt_dataset import filter_flatten_filled_drop_cols

import warnings
import logging
log = logging.getLogger()

@functoolz.curry
def filter_preproc_data(data, target, preprocessor=None):
    X = data.values
    # if preprocessor is None:
    #     preprocessor = MinMaxpreprocessor()
    #     preprocessor.fit(X)
    try:
        X_values = preprocessor.transform(X)
    except NotFittedError:
        raise Exception(f"{preprocessor} not fitted. consider passing fitted preprocessor")

    data = pd.DataFrame(X_values, index=data.index, columns=data.columns)
    data._preprocessor = preprocessor
    return data, target


class SKDataLoader(object):
    """Custom data loaders for scikit-learn"""

    def __init__(self, tgt_file, feat_file, idx_col, tgt_col, 
                 feat_columns=None, time_order_col=None, 
                 category_map=C.DEFAULT_MAP, 
                 fill_value=0., flatten=C.DEFAULT_FLATTEN, 
                 cols_to_drop=C.DEFAULT_DROP_COLS,
                 preprocessor=None):

        self._tgt_file = tgt_file
        self._feat_file = feat_file
        self._idx_col = idx_col
        self._tgt_col = tgt_col
        self._feat_columns = feat_columns
        self._time_order_col= time_order_col
        self._category_map = category_map
        
        self._filter_flatten_filled_drop_cols = filter_flatten_filled_drop_cols(cols_to_drop=cols_to_drop,
                                                                                aggfunc=flatten,
                                                                                fill_value=fill_value)
        self._preprocessor = None
        if preprocessor is not None:
            if isinstance(preprocessor, (list, tuple)):
                self._preprocessor = preprocessor
            else:
                self._preprocessor = [preprocessor]
        return

    def _get_data(self, preprocessor):
        device = 'cpu'
        transform = [ptd.identity_nd]
        filters = []
        if preprocessor is not None:
            for s in preprocessor:
                filters.append(filter_preproc_data(preprocessor=s))
        filters.append(self._filter_flatten_filled_drop_cols)

        dataset = ptd.BaseDataset(self._tgt_file, 
                                  self._feat_file, 
                                  self._idx_col, 
                                  self._tgt_col,
                                  feat_columns=self._feat_columns, 
                                  time_order_col=self._time_order_col,
                                  category_map=self._category_map,
                                  filter=filters,
                                  transform=transform,
                                  device=device
                                 )

        X = dataset.data
        y = dataset.target
        return X, y

    def get_preprocessor(self, refit=False):
        if refit:
            if self._preprocessor is not None:
                preprocessor = []
                for p in reversed(self._preprocessor):
                    X, _ = self._get_data(preprocessor)

                    p.fit(X)
                    preprocessor.insert(0, p)
                self._preprocessor = preprocessor
                log.info("preprocessors fitted")
        return self._preprocessor

    def read_data(self, refit=False):
        preprocessor = self.get_preprocessor(refit=refit)
        X, y = self._get_data(preprocessor)
        return X, y
