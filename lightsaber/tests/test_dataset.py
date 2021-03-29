#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from lightsaber import constants as C
from lightsaber.data_utils import utils as du
from lightsaber.data_utils import pt_dataset as ptd
from lightsaber.data_utils import sk_dataloader as skd

import io


# inline YAML config
_conf = """
tgt_file: /home/shared/expt_covid19/cohorts/severity_v1/SEVERITY_OUT_V1.csv
feat_file: /home/shared/expt_covid19/cohorts/severity_v1/SEVERITY_FEAT_V1_DEMO-COMORB.csv

idx_col: ["EID"]
tgt_col: ["SEVERITY"]
feat_cols: ["SEX", "BMI_LATEST", "AGE", 
    "has_MI", "has_STROKE" , "has_ISCH_STROKE", "has_ASTHMA", "has_RENAL",
    "has_COPD", "has_DEMENTIA", "has_NEURONE", "has_PARKINSON", "has_CANCER", "has_DM"]
"""


def test_pt_dataset():
    conf = du.yaml.load(io.StringIO(_conf), Loader=du._Loader)

    tgt_file = conf['tgt_file']
    feat_file = conf['feat_file']
    category_map = dict()
    device = 'cpu'

    dataset = ptd.BaseDataset(tgt_file,
                              feat_file,
                              idx_col=conf['idx_col'],
                              tgt_col=conf['tgt_col'],
                              feat_columns=conf['feat_cols'],
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
    
    dataloader = ptd.DataLoader(dataset, batch_size=8) #, collate_fn=collate_fn)
    batch = iter(dataloader).next()
    print("Can pass through data loader. bs = 8")
    print("Shapes", batch[0].shape, batch[1].shape, len(batch[2]), len(batch[3]))
    return True


def test_flattened_dataset():
    conf = du.yaml.load(io.StringIO(_conf), Loader=du._Loader)

    tgt_file = conf['tgt_file']
    feat_file = conf['feat_file']
    category_map = dict()
    fill_value=0.
    flatten=['sum', 'max']
    preprocessor = [MinMaxScaler()]
    
    dataloader = skd.SKDataLoader(tgt_file, feat_file, 
                                  idx_col=conf['idx_col'],
                                  tgt_col=conf['tgt_col'],
                                  feat_columns=conf['feat_cols'],
                                  category_map=category_map,
                                  fill_value=fill_value,
                                  flatten=flatten,
                                  preprocessor=preprocessor)

    # Preprocessors before fitting
    preprocessors = dataloader.get_preprocessor(refit=False)
    X, y = dataloader.read_data(refit=True)

    # Preprocessors after fitting
    up_preprocessors = dataloader.get_preprocessor(refit=False)
    print(f"Preprocessors updated: {preprocessors} -> {up_preprocessors}")
    up_dataloader = skd.SKDataLoader(tgt_file, feat_file, 
                                     idx_col=conf['idx_col'],
                                     tgt_col=conf['tgt_col'],
                                     feat_columns=conf['feat_cols'],
                                     category_map=category_map,
                                     fill_value=fill_value,
                                     flatten=flatten,
                                     preprocessor=up_preprocessors)
    up_X, up_y = up_dataloader.read_data(refit=False)
    assert np.allclose(X, up_X)
    assert np.allclose(y, up_y)
    print("Test passed: handling scikit data properly")
    return True


if __name__ == "__main__":
    print("Testing: pytorch access")
    test_pt_dataset()
    print("---------------------------------------------\n")

    print("Testing: flattened access")
    test_flattened_dataset()
    print("---------------------------------------------\n")

    print("All tests passed")

