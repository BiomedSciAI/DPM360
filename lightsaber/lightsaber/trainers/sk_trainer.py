#!/usr/bin/env python
# Author : James Codella<jvcodell@us.ibm.com>, Prithwish Chakraborty <prithwish.chakraborty@ibm.com>
# date   : 2020-06-02
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

import numpy as np
import os
import json
import pickle
from pathlib import Path

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit, KFold
from sklearn.experimental import enable_hist_gradient_boosting

import mlflow
import mlflow.sklearn
import time
import six
import warnings
import uuid

from lightsaber import constants as C
from lightsaber.metrics import Metrics
from lightsaber.trainers import helper
from lightsaber.data_utils import sk_dataloader as skd
from lightsaber.data_utils import utils as du
from lightsaber.trainers.helper import (setup_mlflow, get_model, get_predefined_split,
                                        load_sk_model, save_sk_model, import_model_class)


import logging
log = logging.getLogger()


# ***********************************************************************
#         SK Model Trainer
# ***********************************************************************
class SKModel(object):
    """SKModel
    """
    def __init__(self,
                 base_model,
                 model_params=None,
                 name="undefined_model_name"):
        """
        Parameters
        ----------
        base_model:
            base scikit-learn compatible model (classifier) defining model logic
        model_params:
            if provided, sets the model parameters for base_model
        name:
            name of the model
        """
        super(SKModel, self).__init__()
        self.model = base_model
        if model_params is not None:
            try:
                self.set_params(**model_params)
            except Exception as e:
                warnings.warn("couldnt set model params - base_model/model_params inconsitent with scikit-learn")
                log.debug(f'Error in model params:{e}')
        self.__name__ = name

        self.metrics = {}
        self.proba = []
        # self.params = self.model.get_params()

    @property
    def params(self):
        try:
            params = self.model.get_params()
        except AttributeError:
            raise DeprecationWarning("This is deprecated. will be dropped in v0.3. models should be sklearn compatible i.e. should have get_params. moving forward but this will be inconsistent with tuning")
            params = self.model_params
        return params

    def set_params(self, **parameters):
        self.model.set_params(**parameters)
        return self

    def fit(self, X, y, experiment_name=""): # default exp name is timestamp
        """
        Fits self.model to X, given y.
        Args:
          X (np.array): Feature matrix
          y (np.array): Binary labels for prediction
          experiment_name (str): Name for experiment as defined in config, construction of SKModel object
        Returns np.array predictions for each instance in X.
        """
        self.model.fit(X,y)
        # self.params = self.model.get_params()
        return self

    def predict(self, X):
        """
        Uses model to predict labels given input X.
        Args:
          X (np.array): Feature matrix
        Returns np.array predictions for each instance in X.
        """
        return self.model.predict(X)

    def calibrate(self, X, y):
        ccc = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
        ccc.fit(X, y)
        self.model = ccc
        #  self.params = self.model.get_params()
        return self

    def tune(self,
             X, y,
             hyper_params,
             experiment_name,
             cv=C.DEFAULT_CV,
             scoring=C.DEFAULT_SCORING_CLASSIFIER,
             ):  ## NEEDS MODIFICATION
        """Tune hyperparameters for model. Uses mlflow to log best model, Gridsearch model, scaler, and best_score

        Parameters
        ----------
        X: np.array
            Feature matrix
        y: np.array
            Binary labels for prediction
        hyper_params:  dict
            Dictionary of hyperparameters and values/settings for model's hyperparameters.
        experiment_name: str
            Name for experiment as defined in config, construction of SKModel object
        cv: int or cv fold
            pre-defined cv generator or number
        """
        gs = GridSearchCV(estimator=self.model,
                          cv=cv,
                          param_grid=hyper_params,
                          verbose=2,
                          scoring=scoring)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        self.set_params(**gs.best_params_)
        return self.model, gs

    def predict_proba(self, X):
        """
        Predicts on X and returns class probabilitiyes

        Parameters
        ----------
        X: np.array
            Feature matrix

        Returns
        -------
        array of shape (n_samples, n_classes)
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X,y)

    def predict_patient(self, patient_id, test_dataloader):
        p_X, _ = test_dataloader.get_patient(patient_id)
        return self.predict_proba(p_X)


def run_training_with_mlflow(mlflow_conf:dict,
                             wrapped_model:SKModel,
                             train_dataloader:skd.SKDataLoader,
                             val_dataloader:skd.SKDataLoader=None,
                             test_dataloader:skd.SKDataLoader=None,
                             **kwargs):
    """
    Function to run supervised training for classifcation

    Parameters
    ----------
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    wrapped_model: SKModel
        wrapped SKModel
    train_dataloader: skd.SKDataLoader
        training dataloader
    val_dataloader: skd.SKDataLoader, optional
        validation dataloader
    test_dataloader: skd.SKDataLoader, optional
        test dataloader
    model_path: str, optional
        prefix for storing model in MlFlow
    artifacts: dict, optional
        any artifact to be logged by user
    metrics: Callable, optional
        if specified, used for calculating all metrics. else inferred from problem type
    tune: bool, optional
        if specified tune model based on inner cv. Default: False
    scoring: Callable, optional
        used when tune=True. sklearn compatible scoring function to score the models for grid search. default: C.DEFAULT_SCORING_CLASSIFIER
    inner_cv: object, optional
        used when tune=True. sklearn compatible cross validation folds to for grid search. default: C.DEFAULT_CV
    h_search: dict, optional
        used when tune=True (required). sklearn compatible search space for grid search. 
    run_id: str, optional
        if specified uses existing mlflow run.
    kwargs: dict, optional
        remaining keyword argumennts are used as experiment tags

    Returns
    -------
    tuple:
        (run_id, run_metrics, y_val, y_val_hat, y_val_pred, y_test, y_test_hat, y_test_pred,)
    """
    model_path = kwargs.pop('model_path', 'model')
    artifacts = kwargs.pop('artifacts', dict())

    mlflow_conf.setdefault('problem_type', 'classifier')
    mlflow_setup = setup_mlflow(**mlflow_conf)

    # Support for user-defined run time metrics
    _metrics = kwargs.pop('metrics', None)
    if _metrics is not None:
        calculate_metrics = _metrics
        assert callable(calculate_metrics), f"metric function {_metrics} must be callable."
    else:
        calculate_metrics = Metrics(mlflow_conf['problem_type'])
    log.debug(f"Mlflow setup: {mlflow_setup}")
    log.debug(f"Used metrics: {calculate_metrics}")

    experiment_name = mlflow_setup['experiment_name']
    log.debug(f'Starting experiment {experiment_name}')

    experiment_tags = dict()
    experiment_tags.update(**kwargs)

    # support for tuning
    tune = kwargs.get('tune', False)
    if tune:
        inner_cv = kwargs.get('inner_cv', C.DEFAULT_CV)
        h_search = kwargs.pop('h_search', None)
        if h_search is None:
            raise AttributeError('if tuner is requested, h_search should be provided')
        scoring = kwargs.get('scoring', C.DEFAULT_SCORING_CLASSIFIER)

    # Support for resuming a run
    run_id = kwargs.pop('run_id', None)
    with mlflow.start_run(run_id=run_id):
        run_id = mlflow.active_run().info.run_id
        _start_time = time.time()

        X_train, y_train = train_dataloader.get_data()

        if val_dataloader is not None:
            X_val, y_val = val_dataloader.get_data()
            outer_cv, _X, _y = get_predefined_split(X_train, y_train, X_val, y_val)
        else:
            warnings.warn("This path is untested...use with caution")
            outer_cv = kwargs.get('outer_cv', None)
            if outer_cv is None:
                warnings.warn(f'Neither validation, nor outer_cv provided. using KFold({C.DEFAULT_CV}) to get validation split')
                outer_cv = KFold(C.DEFAULT_CV)
            _X = X_train.values if hasattr(X_train, 'values') else X_train
            _y = y_train.values if hasattr(y_train, 'values') else y_train

        if test_dataloader is not None:
            X_test, y_test = test_dataloader.get_data()

        # mlflow.log_params(wrapped_model.model.get_params())
        if tune:
            m, gs = wrapped_model.tune(X=_X, y=_y,
                                       hyper_params=h_search,
                                       cv=inner_cv,
                                       experiment_name=experiment_name,
                                       scoring=scoring)

            mlflow.sklearn.log_model(m, experiment_name + '_model')
            mlflow.sklearn.log_model(gs, experiment_name + '_GridSearchCV')

            log.info(f"Experiment: {experiment_name} has finished hyperparameter tuning")
            log.info("Hyperparameter search space: " + str(h_search))
            # log params
            mlflow.log_params(wrapped_model.params)
            print(f"Best_params:\n {gs.best_params_}")
        else:
            wrapped_model.fit(X=X_train, y=y_train)#, Xstd = X_train_std)

            mlflow.sklearn.log_model(wrapped_model.model, experiment_name + '_model')
            mlflow.log_params(wrapped_model.params)
            log.info(f"Experiment: {experiment_name} has finished training")

        for split_id, (train_index, val_index) in enumerate(outer_cv.split(_X, _y)):
            if split_id >= 1:
                warnings.warn("Current logic for tune and implicit outer_cv not correct... skipping calculating metrics for validation fold")
                y_val = None
                y_val_hat = None
                y_val_pred = None
                val_score = None
                break

            _, _X_val = _X[train_index, :], _X[val_index, :]
            _, y_val = _y[train_index], _y[val_index]

            y_val_pred = wrapped_model.predict_proba(_X_val)
            # if y_val_pred.ndim > 1:
            #     y_val_pred = y_val_pred[:, 1]

            y_val_hat = wrapped_model.predict(_X_val)
            val_score = wrapped_model.score(_X_val, y_val)
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val.squeeze(axis=1)

        if test_dataloader is not None:
            y_test_pred = wrapped_model.predict_proba(X_test)
            # if y_test_pred.ndim > 1:
            #     y_test_pred = y_test_pred[:, 1]
            y_test_hat = wrapped_model.predict(X_test)
            test_score = wrapped_model.score(X_test, y_test)
            if y_test.ndim == 2 and y_test.shape[1] == 1:
                y_test = y_test.squeeze(axis=1)
        else:
            y_test=None
            y_test_hat=None
            y_test_pred=None
            test_score =None

        _end_time = time.time()
        run_time = (_end_time - _start_time)

    # Experiment ended. logging things
    experiment_tags.update(dict(run_time=run_time))

    # Calculate metrics
    try:
        run_metrics = calculate_metrics(y_val=y_val, y_val_hat=y_val_hat, y_val_proba=y_val_pred, val_score=val_score,
                                        y_test=y_test, y_test_hat=y_test_hat, y_test_proba=y_test_pred, test_score=test_score)

    except Exception as e:
        warnings.warn(f"{e}")
        log.warning(f"something went wrong while computing metrics: {e}")
        run_metrics = None

    # log metrics
    helper.log_metrics(run_metrics, run_id=run_id)

    # Other artifacts
    _tmp = {f"artifact/{art_name}": art_val
            for art_name, art_val in six.iteritems(artifacts)}
    helper.log_artifacts(_tmp, run_id=run_id, mlflow_uri=mlflow_setup['mlflow_uri'], delete=True)

    helper.set_tags(experiment_tags, run_id=run_id)

    return (run_id, 
            run_metrics, 
            y_val, y_val_hat, y_val_pred,
            y_test, y_test_hat, y_test_pred,
            )


def load_model_from_mlflow(run_id,
                           mlflow_conf,
                           wrapped_model=None,
                           model_path="model",
                           ):
    """Method to load a trained model from mlflow

    Parameters
    ----------
    run_id: str
        mlflow run id for the trained model
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    wrapped_model: SKModel
        model architecture to be logged
    model_path: str
        output path where model checkpoints are logged

    Returns
    -------
    SKModel:
        wrapped model with saved weights and parameters from the run
    """
    mlflow_setup = helper.setup_mlflow(**mlflow_conf)
    model_uri = f"runs:/{run_id}/{mlflow_setup['experiment_name']}_{model_path}"
    run_data = helper.fetch_mlflow_run(run_id,
                                       mlflow_uri=mlflow_setup['mlflow_uri'],
                                       parse_params=True
                                       )

    hparams = run_data['params']

    # ckpt_path = helper.get_artifact_path(run_data['artifact_paths'][0],
    #                                      artifact_uri=run_data['info'].artifact_uri)
    #  wrapped_model = load_model(wrapped_model, ckpt_path)
    model_name = run_data['tags']['model']
    if wrapped_model is None:
        base_model = mlflow.sklearn.load_model(model_uri)
        wrapped_model = SKModel(base_model, hparams, name=model_name)
    return wrapped_model


def register_model_with_mlflow(run_id,
                               mlflow_conf,
                               wrapped_model=None,
                               registered_model_name=None,
                               model_path='model',
                               **artifacts
                               ):
    """Method to register a trained model

    Parameters
    ----------
    run_id: str
        mlflow run id for the trained model
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    wrapped_model: SKModel, optional
        model architecture to be logged. If not provided, the model is directly read from mlflow
    registered_model_name: str
        name for registering the model
    model_path: str
        output path where model will be logged
    artifacts: dict
        dictionary of objects to log with the model
    """
    # Getting run info
    mlflow_setup = helper.setup_mlflow(**mlflow_conf)
    wrapped_model = load_model_from_mlflow(run_id, mlflow_conf,
                                           wrapped_model=wrapped_model, model_path=model_path)

    if registered_model_name is None:
        model_name = wrapped_model.__name__
        registered_model_name = f"{mlflow_setup['experiment_name']}_{model_name}_v{uuid.uuid3()}"

    # Registering model
    with mlflow.start_run(run_id):
        try:
            mlflow.sklearn.log_model(wrapped_model.model, model_path, registered_model_name=registered_model_name)
        except Exception as e:
            log.error(f'Exception during logging model: {e}. Continuing to dump artifacts')

    # logging other artifacts
    dumper = helper.model_register_dumper(registered_model_name=registered_model_name)
    helper.log_artifacts(artifacts, run_id, mlflow_uri=mlflow_setup['mlflow_uri'], dumper=dumper, delete=True)
    return
