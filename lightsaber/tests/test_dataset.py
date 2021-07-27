#!/usr/bin/env python3
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
# limitations under the License.from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from lightsaber import constants as C
from lightsaber.data_utils import utils as du
from lightsaber.data_utils import pt_dataset as ptd
from lightsaber.data_utils import sk_dataloader as skd

import io
import os

# inline YAML config
_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                        '..', 'examples', 'in_hosptial_mortality', 'data'))
print(_data_dir, os.path.isdir(_data_dir))
_conf = """
# USER DEFINED
tgt_col: y_true
idx_cols: stay
time_order_col: 
    - Hours
    - seqnum

feat_cols: null

train:
    tgt_file: '{DATA_DIR}/IHM_V0_COHORT_OUT_EXP-SPLIT0-train.csv'
    feat_file: '{DATA_DIR}/IHM_V0_FEAT_EXP-SPLIT0-train.csv'

val:
    tgt_file: '{DATA_DIR}/IHM_V0_COHORT_OUT_EXP-SPLIT0-val.csv'
    feat_file: '{DATA_DIR}/IHM_V0_FEAT_EXP-SPLIT0-val.csv'

test:
    tgt_file: '{DATA_DIR}/IHM_V0_COHORT_OUT_EXP-SPLIT0-test.csv'
    feat_file: '{DATA_DIR}/IHM_V0_FEAT_EXP-SPLIT0-test.csv'

# DATA DEFINITIONS
category_map:
  Capillary refill rate: ['0.0', '1.0']
  Glascow coma scale eye opening: ['To Pain', '3 To speech', '1 No Response', '4 Spontaneously',
                                   'To Speech', 'Spontaneously', '2 To pain', 'None'] 
  Glascow coma scale motor response: ['1 No Response' , '3 Abnorm flexion' , 'Abnormal extension' , 'No response',
                                      '4 Flex-withdraws' , 'Localizes Pain' , 'Flex-withdraws' , 'Obeys Commands',
                                      'Abnormal Flexion' , '6 Obeys Commands' , '5 Localizes Pain' , '2 Abnorm extensn']
  Glascow coma scale total: ['11', '10', '13', '12', '15', '14', '3', '5', '4', '7', '6', '9', '8']
  Glascow coma scale verbal response: ['1 No Response', 'No Response', 'Confused', 'Inappropriate Words', 'Oriented', 
                                       'No Response-ETT', '5 Oriented', 'Incomprehensible sounds', '1.0 ET/Trach', 
                                       '4 Confused', '2 Incomp sounds', '3 Inapprop words']

numerical: ['Heart Rate', 'Fraction inspired oxygen', 'Weight', 'Respiratory rate', 
            'pH', 'Diastolic blood pressure', 'Glucose', 'Systolic blood pressure',
            'Height', 'Oxygen saturation', 'Temperature', 'Mean blood pressure']

normal_values:
  Capillary refill rate: 0.0
  Diastolic blood pressure: 59.0
  Fraction inspired oxygen: 0.21
  Glucose: 128.0
  Heart Rate: 86
  Height: 170.0
  Mean blood pressure: 77.0
  Oxygen saturation: 98.0
  Respiratory rate: 19
  Systolic blood pressure: 118.0
  Temperature: 36.6
  Weight: 81.0
  pH: 7.4
  Glascow coma scale eye opening: '4 Spontaneously'
  Glascow coma scale motor response: '6 Obeys Commands'
  Glascow coma scale total:  '15'
  Glascow coma scale verbal response: '5 Oriented'
"""

def test_pt_dataset():
    expt_conf = du.yaml.load(_conf.format(DATA_DIR=_data_dir),
                             Loader=du._Loader)

    tgt_file = expt_conf['val']['tgt_file']
    feat_file = expt_conf['val']['feat_file']
    category_map = expt_conf['category_map']
    device = 'cpu'

    dataset = ptd.BaseDataset(tgt_file,
                              feat_file,
                              idx_col=expt_conf['idx_cols'],
                              tgt_col=expt_conf['tgt_col'],
                              feat_columns=expt_conf['feat_cols'],
                              category_map=category_map,
                              device=device)
    print("Can create dataset")

    idx = 2
    data = dataset[idx]

    (input_dim, output_dim) = dataset.shape
    print("Can access and generate shapes")
    print("Shape", input_dim, output_dim)
    print("Output", data)

    dataloader = ptd.DataLoader(dataset, batch_size=1) # , collate_fn=collate_fn)
    batch = iter(dataloader).next()
    print("Can pass through data loader. bs = 1")
    print("Shapes", batch[0].shape, batch[1].shape, len(batch[2]), len(batch[3]))
    
    dataloader = ptd.DataLoader(dataset, batch_size=8, collate_fn=ptd.collate_fn)
    batch = iter(dataloader).next()
    print("Can pass through data loader. bs = 8")
    print("Shapes", batch[0].shape, batch[1].shape, len(batch[2]), len(batch[3]))
    return True


def test_flattened_dataset():
    expt_conf = du.yaml.load(_conf.format(DATA_DIR=_data_dir),
                             Loader=du._Loader)

    tgt_file = expt_conf['val']['tgt_file']
    feat_file = expt_conf['val']['feat_file']
    category_map = expt_conf['category_map']
    
    flatten = 'sum'
    preprocessor = StandardScaler()
    train_filter = [ptd.filter_preprocessor(cols=expt_conf['numerical'], 
                                            preprocessor=preprocessor,
                                            refit=True),
                   ]

    first_dataloader = skd.SKDataLoader(tgt_file=expt_conf['val']['tgt_file'],
                                        feat_file=expt_conf['val']['feat_file'],
                                        idx_col=expt_conf['idx_cols'],
                                        tgt_col=expt_conf['tgt_col'],
                                        feat_columns=expt_conf['feat_cols'],
                                        time_order_col=expt_conf['time_order_col'],
                                        category_map=expt_conf['category_map'],
                                        filter=train_filter,
                                        fill_value=expt_conf['normal_values'],
                                        flatten=flatten,
                                       )

    # Preprocessors before fitting
    X, y = first_dataloader.get_data()
    

    # Preprocessors after fitting
    fitted_filter = [ptd.filter_preprocessor(cols=expt_conf['numerical'], 
                                             preprocessor=preprocessor, refit=False),
                                      ]
    up_dataloader = skd.SKDataLoader(tgt_file=expt_conf['val']['tgt_file'],
                                        feat_file=expt_conf['val']['feat_file'],
                                        idx_col=expt_conf['idx_cols'],
                                        tgt_col=expt_conf['tgt_col'],
                                        feat_columns=expt_conf['feat_cols'],
                                        time_order_col=expt_conf['time_order_col'],
                                        category_map=expt_conf['category_map'],
                                        filter=fitted_filter,
                                        fill_value=expt_conf['normal_values'],
                                        flatten=flatten,
                                       )
    up_X, up_y = up_dataloader.get_data()
    assert np.allclose(X, up_X)
    assert np.allclose(y, up_y)
    print("Test passed: handling scikit data properly")
    return True

def test_feat_seletion():
    """docstring for test_feat_seletion"""
    import pandas as pd
    df = pd.DataFrame(columns = ['CCS_128', 'CCS', 'AGE', 'GENDER'])

    feat_columns = ['CCS_.*', 'AGE'] 
    selected_features = ptd.BaseDataset._select_features(df, feat_columns)
    expected_features = ['CCS_128', 'AGE']
    assert set(selected_features) == set(expected_features)

    feat_columns = ['CCS', 'AGE'] # would select 'CCS' and 'AGE'
    selected_features = ptd.BaseDataset._select_features(df, feat_columns)
    expected_features = ['CCS', 'AGE']
    assert set(selected_features) == set(expected_features)
    return
    

if __name__ == "__main__":
    tests_failed = 0
    err_msg = ""
    
    print("Testing: feature selection")
    try:
        test_feat_seletion()
    except Exception as e:
        tests_failed += 1
        err_msg += f'Test failed: feature selection. Error:\n{e}'
    print("---------------------------------------------\n")

    print("Testing: pytorch access")
    try:
        test_pt_dataset()
    except Exception as e:
        tests_failed += 1
        err_msg += f'Test failed: pytorch access. Error:\n{e}'
    print("---------------------------------------------\n")

    print("Testing: flattened access")
    try:
        test_flattened_dataset()
    except Exception as e:
        tests_failed += 1
        err_msg += f'Test failed: flattened access. Error:\n{e}\n'
    print("---------------------------------------------\n")

    if tests_failed == 0:
        print("All tests passed")
    else:
        print(f'Tests failed: {tests_failed}')
        print(err_msg)
