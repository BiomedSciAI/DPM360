#!/usr/bin/env python

import argparse
import sys
import os
import pickle
import numpy as np
import mlflow

from sklearn.model_selection import PredefinedSplit

from lightsaber import constants as C


# ***********************************************************************
#         General Utils
# ***********************************************************************
def import_model_class(name):
    components = name.split('.')
    base = ""
    for comp in components[0:-1]:
        base = base + "." + comp
    base = base[1:]
    print(base)
    mod = __import__(base, fromlist=[components[-1]])
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
def get_experiment_name(conf):
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
    X = X_train.append(X_val) if X_val is not None else X_train
    y = np.append(y_train,y_val) if y_val is not None else y_train

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in X_train.index else 0 for x in X.index]
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




