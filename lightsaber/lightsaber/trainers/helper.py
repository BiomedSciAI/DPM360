#!/usr/bin/env python
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
import argparse
import sys
import os
from pathlib import Path
import six
from tempfile import NamedTemporaryFile
import shutil
import pickle
import numpy as np
import mlflow
from toolz import functoolz
import warnings
import importlib

from sklearn.model_selection import PredefinedSplit

from lightsaber import constants as C

import logging
log = logging.getLogger()

# ***********************************************************************
#         General Utils
# ***********************************************************************
def import_model_class(name):
    components = name.split('.')
    base = ""
    for comp in components[0:-1]:
        base = base + "." + comp
    base = base[1:]
    log.info(f'Loading {components[-1]} from {base}')
    mod = importlib.__import__(base, fromlist=[components[-1]])
    mod = getattr(mod, components[-1])
    return mod


def get_model(model_type, model_params=None):
    """generate a model from model type and parameters

    Parameters
    ----------
    model_type : importable model name
    model_params : model parameters

    Returns
    -------
    instantiated model
    """
    model_to_call = import_model_class(model_type)
    if model_params is not None:
        model = model_to_call(**model_params)
    else:
        model = model_to_call(verbose=2)
    return model

# ***********************************************************************
#         MLFlow
# ***********************************************************************
def get_experiment_name(**conf):
    experiment_name = '{}-{}-{}-{}'.format(os.getlogin(), 
                                           conf['problem_type'],
                                           conf['model_name'], 
                                           conf['experiment_keyword'])
    return experiment_name


def setup_mlflow(mlflow_uri=C.MLFLOW_URI,
                 experiment_name=None,
                 problem_type=C.PROBLEM_TYPE,
                 model_name=C.MODEL_NAME, 
                 experiment_keyword=C.EXPERIMENT_KEYWORD, 
                 **kwargs):

    mlflow.set_tracking_uri(mlflow_uri)
    
    if experiment_name is None:
        experiment_name = get_experiment_name(problem_type=problem_type,
                                              model_name=model_name,
                                              experiment_keyword=experiment_keyword)
    mlflow.set_experiment(experiment_name)

    return dict(problem_type=problem_type, experiment_name=experiment_name, mlflow_uri=mlflow_uri)

def fetch_mlflow_run(run_id, 
                     mlflow_uri=C.MLFLOW_URI,
                     artifacts_prefix=['model']):
    # ref: https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    run = client.get_run(run_id)
    info = run.info
    data = run.data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = []
    for _prefix in artifacts_prefix:
        artifacts += [f.path for f in client.list_artifacts(run_id, _prefix)]
    return dict(info=info,
                params=data.params,
                metrics=data.metrics,
                tags=tags, 
                artifact_paths=artifacts)

def fetch_mlflow_experiment(experiment_name, 
                            mlflow_uri=C.MLFLOW_URI,
                            **kwargs):
    # client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    run_df = mlflow.search_runs(experiment_id, **kwargs)
    return run_df

def get_artifact_path(artifact_path,
                      artifact_uri,
                      ): 
    artifact_path = Path(artifact_uri.lstrip('file:')) / str(artifact_path)
    assert artifact_path.is_file()
    return str(artifact_path)


def safe_dumper(obj, obj_name):
    """
    If obj is not a file name dump it as a file. 
    """
    is_file = False
    try:
        is_file = Path(obj).is_file()
    except Exception as e:
        warnings.warn(f"problem during file checking: {e}\ncontinuing")

    if not is_file:
        assert not hasattr(obj, 'read'), "Object cannot be a open stream"
        with NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(obj, open(tmp_file.name, "wb"))
            tmp_filename = tmp_file.name
    else:
        tmp_filename = obj
    art_name = obj_name
    return tmp_filename, obj_name


@functoolz.curry
def model_register_dumper(obj, obj_name, registered_model_name=None):
    """
    If obj is not a file name dump it as a file. 
    """
    if (obj_name == 'test_feat_file'):
        temp_filename = 'X_test.csv'
    elif (obj_name == 'test_tgt_file'):
        temp_filename = 'y_test.csv'
    elif (obj_name == 'config'):
        temp_filename = registered_model_name+'.yaml'
    else:
        temp_filename = None
    if temp_filename is not None:
        shutil.copy(obj, temp_filename)
    art_name =  "features"
    return temp_filename, art_name


def log_artifacts(artifacts, 
                  run_id, 
                  mlflow_uri=C.MLFLOW_URI, 
                  dumper=safe_dumper, 
                  delete=False):
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    
    ret_fnames = []
    for art_name, art_val in six.iteritems(artifacts):
        tmp_filename, art_name = dumper(art_val, art_name)
        if tmp_filename is not None:
            client.log_artifact(run_id, tmp_filename, f"{art_name}")
        ret_fnames.append(tmp_filename)
        log.info(f"Logged {art_name} to {ret_fnames[-1]}") 
        if delete: 
            os.remove(tmp_filename)
    return ret_fnames

# ***********************************************************************
#         SK Model Utils
# ***********************************************************************
def sk_parse_args():
    ap = argparse.ArgumentParser('program')
    ap.add_argument("-n", "--tune", dest='tune', required=False, action='store_true', help = "flag if tuning")
    ap.add_argument("-t", "--train", dest='train', required=False, action='store_true', help = "flag if train")
    ap.add_argument("-e", "--eval", dest="evaluate", required=False, action='store_true', help="flag if testing")
    ap.add_argument("-c", "--config", metavar="config", required=True, type=str, help= "config file path")
    ap.add_argument('-l', "--load", metavar="load", required=False, default=None, help= "path/file to SKModel pkl for loading")
    ap.add_argument('-s', "--save", metavar="save", required = False, default=None, help= "path/file to save SKModel pkl")
    ap.add_argument('-pp', "--proba",dest="predict_proba", required=False, type=str, help="flag to output prediction probs")
    ap.add_argument('-pd', "--predict",dest="predict", required=False, type=str, help="flag to output predicted classes")
    ap.add_argument('-cal', "--calibrate", dest='calibrate', required=False, action='store_true', help="flag if calibration should occurs")

    ap.add_argument
    arg = ap.parse_args()
    return arg


def get_predefined_split(X_train, y_train, X_val=None, y_val=None):
    # Using PredefinedSplits to separate training from validation
    # First merge data
    X = np.vstack((X_train, X_val)) if X_val is not None else X_train
    y = np.vstack((y_train, y_val)) if y_val is not None else y_train

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1] * len(X_train) 
    if X_val is not None:
        split_index += [0] * len(X_val)

    pre_split = PredefinedSplit(test_fold=split_index)
    return pre_split, X, y

# De-prioritized merge from t2e
# def get_predefined_split(X_train, y_train, X_valid, y_valid):
#     # Reference: https://www.wellformedness.com/blog/using-a-fixed-training-development-test-split-in-sklearn/
#     X = np.concatenate([X_train, X_valid])
#     y = np.concatenate([y_train, y_valid])
#     test_fold = np.concatenate([
#         np.full(-1, X_train.shape[1]),
#         np.zeros(X_valid.shape[1])
#     ])
#     ps = PredefinedSplit(test_fold)
#     return X, y, ps


def save_sk_model(skmodel, model_path): # simple pickle implementation placeholder for model saver
    with open(model_path, 'wb') as fid:
        pickle.dump(skmodel, fid)


def load_sk_model(model_path): # simple pickle implementation placeholder for model loader
    if 'mlflow' in model_path:  #load from mlflow
        sk_model = mlflow.sklearn.load_model(model_path)
    else: # load from pkl file
        with open(model_path, 'rb') as fid:
            sk_model = pickle.load(fid)
    return sk_model
