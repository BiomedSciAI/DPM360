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

import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
import os
import pickle
from tqdm import tqdm
from lightsaber import constants as C

import logging
log = logging.getLogger()

try:
    from ruamel import yaml
    _Loader = yaml.RoundTripLoader
    _Dumper = yaml.RoundTripDumper
except ImportError:
    log.warning("Couldnt import ruamel. falling back to yaml")
    import yaml
    _Loader = yaml.SafeLoader
    _Dumper = yaml.SafeDumper


class ConfReader(object):
    """Simple configuration reader"""

    def __init__(self, fname):
        """Class to read the experiment configuration

        Parameters
        ----------
        fname : TODO
        """
        self._fname = fname
        self.name = os.path.basename(fname).rstrip('.yml')
        with open(self._fname) as fIn:
            self.config = yaml.load(fIn, Loader=_Loader)
        return
        

def create_splits(idx, y, c_prop=0.05, v_prop=0.15, t_prop=0.1, random_state=None):
    n_examples = len(idx)
    t_size = int(n_examples * t_prop)
    c_size = int(n_examples * c_prop)
    v_size = int(n_examples * v_prop)

    _train, test = train_test_split(idx, test_size=t_size, stratify=y, random_state=random_state)
    _train, cal = train_test_split(_train, test_size=c_size,
                                   stratify=y.loc[_train], random_state=random_state)
    train, val = train_test_split(_train, test_size=v_size,
                                  stratify=y.loc[_train], random_state=random_state)

    msg = """Summary:
    train:\t {}/{:0.2f}\%
    validation:\t {}/{:0.2f}\%
    calibration:\t {}/{:0.2f}\%
    test:\t {}/{:0.2f}\%
    """.format(len(train), (100.0 * len(train)) / (n_examples),
               len(val), (100.0 * len(val)) / (n_examples),
               len(cal), (100.0 * len(cal)) / (n_examples),
               len(test), (100.0 * len(test)) / (n_examples)
               )
    print(msg)
    return train, val, cal, test


def get_stratifier(tgt_data, tgt_col, pat_idx, strategy='count',
                   upper_bound=0.95):
    y = tgt_data.reset_index().groupby(pat_idx).agg({tgt_col: strategy})
    _upp_bound = y[tgt_col].quantile(upper_bound)
    y.loc[(y[tgt_col] > _upp_bound), tgt_col] = _upp_bound
    return y
    

def generate_split_indices(tgt_data, cv_file, tgt_col, random_state=None, 
                           prefix=None, split_id=1, **kwargs):
    if os.path.isfile(cv_file):
        try:
            log.info("reading from existing file")
            with open(cv_file, 'rb') as fIn:
                data = pickle.load(fIn)
            return data
        except Exception as e:
            log.warning('Problem reading from existing CV file. generating a new one. Original error:{}'.format(e))
            pass
        
    log.info("generating new splits")  
    Y = tgt_data[tgt_col]
    idx = tgt_data.index

    if hasattr(idx, 'names'):
        indexer = idx.names
    else:
        indexer = idx.name

    (train, val, cal, test) = create_splits(idx, Y, random_state=random_state)

    if not os.path.isdir(os.path.dirname(cv_file)):
        os.makedirs(os.path.dirname(cv_file))
    with open(cv_file, 'wb') as fOut:
        data = dict(indexer=indexer,
                    train=train,
                    val=val,
                    cal=cal,
                    test=test)
        pickle.dump(data, fOut)
    log.debug("splits outputted to {}".format(cv_file))
    return data


# def save_data_to_dir(tgt_data, feat_data, tgt_col, split_idx, 
#                      time_order_col=TIME_ORDER_COL, expt_dir=None, 
#                      prefix=None, split_id=1, **kwargs):
#     tgt_name = kwargs.get('tgt_name', 'OUT')
#     midfix = kwargs.get('midfix', '')
#     print(tgt_name, midfix, prefix)
# 
#     MIDX = pd.IndexSlice
# 
#     if expt_dir is None:
#         expt_dir = os.getcwd()
# 
#     for segment in ['train', 'val', 'cal', 'test']:
#         # segment_dir = os.path.join(expt_dir, segment)
#         # if not os.path.isdir(segment_dir):
#         #     os.makedirs(segment_dir)
# 
#         log.debug(f"Saving data for {segment} to {expt_dir}")
#         if tgt_data is not None:
#             segment_y = tgt_data.loc[MIDX[split_idx[segment]], tgt_col]
#             _fname = f'{prefix}COHORT_{tgt_name}_EXP{midfix}-SPLIT{split_id}-{segment}.csv'
#             segment_y.to_csv(os.path.join(expt_dir, _fname), header=True)
# 
#         if feat_data is not None:
#             segment_x = feat_data.loc[MIDX[split_idx[segment]], :]
#             _fname = f'{prefix}FEAT_EXP{midfix}-SPLIT{split_id}-{segment}.csv'
#             segment_x.to_csv(os.path.join(expt_dir, _fname), header=True)
# 
#         # **Following part is very slow**
#         # for idx in tqdm(split_idx[segment]):
#         #     data = feat_data.loc[idx]
#         #     if type(idx) is tuple or type(idx) is list:
#         #         file_suffix = "-".join([str(x) for x in idx])
#         #     else:
#         #         file_suffix = str(idx)
#         #     fname = os.path.join(segment_dir, f'feat_{file_suffix}.csv')
#         #     data.to_csv(fname, header=True)
#     return

