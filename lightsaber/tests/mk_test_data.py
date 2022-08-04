#!/usr/bin/env python
_desc_doc = """
File to create mock datasets for testing

Data contains 4 attributes: id, time, treatment (A), cov1 (X)

::latex::
    A \sim Binomial(\text{invLogit}(X_{t-1} - \bar{x})/10 - A_{t-1})
    X \sim Normal(A_t + X_{t-1}, 1)

A shifted dataset is also generated with previous values of past covariates.  
Attributes:  
    - independent: id, time, prev_treat, prev_cov1
    - dependent: treatment (classification), prev_cov1 (regression)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from toolz import functoolz
from sklearn.model_selection import train_test_split

import sys
import argparse
import logging
log = logging.getLogger()

# CONSTANTS
EPS = np.finfo(float).eps
EXPT_NAME = 'easiest_sim'
NUM_TIMEPOINTS = 50
NUM_SAMPLES = 1000
SEED = 20

invLogit = lambda x: (np.exp(x) / (1 + np.exp(x)))      # same as sigmoid function (-\inf, +\inf) -> (0, 1) 
fOut = Path(__file__).absolute().parent / 'data'/ f'{EXPT_NAME}.csv'
fOut_shifted = fOut.parent / f'{EXPT_NAME}_shifted.csv'

# SAMPLES
def generate_sample(rep):
    A = np.empty(rep)
    X = np.empty(rep)
    A[0] = np.random.binomial(1, 0.5)
    X[0] = np.random.normal(A[0])

    min_x = X[1]
    mean_x = X[1]
    mean_A = A[1]

    for i in range(1, rep):
        A[i] = np.random.binomial(1, invLogit((X[i-1] - mean_x) / 10 - A[i-1]))
        X[i] = np.random.normal(A[i] + X[i-1])

        mean_A = (mean_A * (i - 1) + A[i]) / i
        mean_x = (mean_x * (i - 1) + X[i]) / i
        min_x = min(min_x, X[i])
    time = np.arange(rep)
    sample = np.transpose(np.vstack((time, A, X)))
    return sample

@functoolz.curry
def create_shifted_targets(df, shift_index,
                           shift_suffix='_history'
                           #shift_columns=['treatment', 'cov1']
                           ):
    tmp = pd.merge(df[[shift_index]], df,how='cross', suffixes=('', shift_suffix))
    tmp = tmp[tmp[f"{shift_index}"] >= tmp[f"{shift_index}{shift_suffix}"]]  
    tmp.set_index([f"{shift_index}", f"{shift_index}{shift_suffix}"], inplace=True)
    return tmp

def create_train_validation_splits(f_shifted, 
                                   idx_cols=['id', 'time'],
                                   feat_cols=['prev_treat', 'prev_cov1'],
                                   tgt_cols=['treatment', 'cov1'],
                                   time_order_col=['time_history']
                                  ):
    data_shifted = pd.read_csv(f_shifted)
    data_shifted.set_index(idx_cols, inplace=True)
    
    # covariates = ['prev_treat', 'prev_cov1']
    # tgt_columns = ['treatment', 'cov1']

    # Splitting dataframe to target and feature such that for each index in target, there is a history of covariates for feature
    group_idx, shift_idx = idx_cols
    
    tgt_df = data_shifted[tgt_cols]
    feat_df = (data_shifted[feat_cols].reset_index()
               .groupby(group_idx)
               .apply(create_shifted_targets(shift_index=shift_idx))
              )[feat_cols]
    feat_df.sort_index(inplace=True)
    tgt_df.to_csv(fOut.parent / f'{EXPT_NAME}_shifted_TGT.csv')
    feat_df.to_csv(fOut.parent / f'{EXPT_NAME}_shifted_FEAT.csv')
    log.info('Shifted data split into target and feat')

    # -------------------------------------------------
    # Splitting the data into segments
    # -------------------------------------------------
    split_size = 0.1
    _train, test = train_test_split(tgt_df.index, 
                                    test_size=split_size, stratify=tgt_df['treatment'])
    train, val = train_test_split(_train, test_size=split_size,
                                  stratify=tgt_df['treatment'].loc[_train])
    log.debug(f'Splits created. #Train: {len(train)} #Val: {len(val)} #Test: {len(test)}')
    split_idx = dict(train=train, test=test, val=val)
    
    # Setting up the feature df for split
    feat_df = feat_df.reset_index().set_index(idx_cols)
    for segment in ['train', 'val', 'test']:   
        log.info(f"creating split for {segment}")
        segment_y = tgt_df.loc[pd.IndexSlice[split_idx[segment]], tgt_cols]
        _fname = fOut.parent / f'{EXPT_NAME}_shifted_TGT_{segment}.csv'
        segment_y.to_csv(_fname, header=True)

        segment_x = feat_df.loc[pd.IndexSlice[split_idx[segment]], :]
        _fname = fOut.parent / f'{EXPT_NAME}_shifted_FEAT_{segment}.csv'
        segment_x.to_csv(_fname, header=True)
        log.debug(f"Target Sizes for {segment}: {segment_y.shape, segment_x.shape}")
    return

def parse_args():
    f"""{_desc_doc}"""
    ap = argparse.ArgumentParser('mk_test_data')
    # Main options
    ap.add_argument("-s", "--seed", metavar='SEED', required=False,
                    type=int, default=SEED,
                    help="seed value to sample the data")
    ap.add_argument("-t", "--num_timepoints", metavar='NUM_TIMEPOINTS', required=False,
                    type=int, default=NUM_TIMEPOINTS,
                    help="Number of timepoints to sample from")
    ap.add_argument("-n", "--num_samples", metavar='NUM_SAMPLES', required=False,
                    type=int, default=NUM_TIMEPOINTS,
                    help="Number of samples")
    # Log options
    ap.add_argument('-v', '--verbose', action="store_true",
                    help="Log option: Write debug messages.")

    arg = ap.parse_args()
    return arg

def init_logs(arg, log):
    if arg and vars(arg).get('verbose', False):
        l = logging.DEBUG
    else:
        l = logging.INFO
    
    # printing to stdout
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(l)
    return


def main():
    arg = parse_args()
    init_logs(arg, log)

    # Grabbing the hyper-params
    seed = arg.seed
    num_timepoints = arg.num_timepoints
    num_samples = NUM_SAMPLES

    np.random.seed(seed)
    data = np.vstack([generate_sample(num_timepoints)  # generate data for each sample
                      for i in range(num_samples)])

    # Massaging data into proper dataframe
    data = pd.DataFrame(data, columns=['time', 'treatment', 'cov1'])
    data['id'] = np.repeat(np.arange(num_samples), num_timepoints)
    data = (data[['id', 'time', 'treatment', 'cov1']]
            .astype({'treatment': int})     # converting treatment to integer
           )

    if not fOut.parent.is_dir():
        fOut.parent.mkdir(parents=True)
    data.to_csv(fOut, index=False)
    log.info(f"Sample genereated to {fOut}\tShape: ({NUM_SAMPLES} x {NUM_TIMEPOINTS}) x 4")
    
    # generating shifted sampled data
    data['prev_cov1'] = data.groupby('id').cov1.shift(1)
    data['prev_treat'] = data.groupby('id').treatment.shift(1)

    data = data[['id', 'time', 'prev_treat', 'prev_cov1', 'treatment', 'cov1']]
    data.set_index(['id', 'time'], inplace=True)
    data = data.loc[pd.IndexSlice[:, 1:], :]
    data.to_csv(fOut_shifted, index=True)
    log.info(f"Shifted Sample genereated to {fOut}\tShape: ({NUM_SAMPLES} x {NUM_TIMEPOINTS}) x 6")
    
    create_train_validation_splits(fOut_shifted)
    log.info(f"Shifted and splitted samples outputted to {fOut.parent})")


if __name__ == "__main__":
    main()
