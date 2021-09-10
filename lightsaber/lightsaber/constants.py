#!/usr/bin/env python3
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
import torch

PAD = -1000
DEFAULT_MAP = dict()
DEFAULT_FLATTEN = 'sum'
DEFAULT_CV = 5
DEFAULT_SCORING_CLASSIFIER = 'roc_auc'

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

_deprecation_warn_msg = "Function/module depcreated. will be dropped in version 0.3"
