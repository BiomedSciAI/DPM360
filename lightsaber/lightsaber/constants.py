#!/usr/bin/env python3
import os
import torch

PAD = -1000
DEFAULT_MAP = dict()
DEFAULT_FLATTEN = 'sum'
DEFAULT_CV = 5

DEFAULT_DROP_COLS = ['INDEX_CLAIM_THRU_DT', 'INDEX_CLAIM_ORDER', 'CLAIM_NO', 'CLM_ADMSN_DT', 'CLM_THRU_DT', 'CLAIM_ORDER']

# MLFLOW config
MLFLOW_URI = os.environ.get('MLFLOW_URI', f"file:{os.path.join(os.path.abspath(os.getcwd()), 'mlruns')}")
PROBLEM_TYPE = "classifier"
MODEL_NAME = 'TestModel'
EXPERIMENT_KEYWORD = "v0" # keyword for experiment for your reference, e.g.: L1 norm

# Labels 
LABEL_POSITIVE = 1
LABEL_NEGATIVE = -1

# Survival
DEFAULT_CENSORED_NAME = 'event'
# DEFAULT_TASK_NAME = 'task'
LABEL_CENSORED = 0
LABEL_OBSERVED = 1


ACT_FUNC_DICT = {
	'ReLU': torch.nn.ReLU(),
	'Tanh': torch.nn.Tanh(),
	'LeakyReLU': torch.nn.LeakyReLU(negative_slope=0.001),
	'Tanhshrink': torch.nn.Tanhshrink(),
	'Hardtanh': torch.nn.Hardtanh(),
	'Softmax': torch.nn.Softmax(),
	'Sigmoid': torch.nn.Sigmoid(),
}

PYTORCH_CLASS_DICT = {
    'LSTM': torch.nn.LSTM,
    'GRU': torch.nn.GRU,
}
