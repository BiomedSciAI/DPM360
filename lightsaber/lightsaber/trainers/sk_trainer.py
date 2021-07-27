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
    def __init__(self, 
                 base_model,
                 model_params=None, 
                 name="undefined_model_name"):
        super(SKModel, self).__init__()
        self.model = base_model
        try:
            self.set_params(**model_params)
        except Exception as e:
            warnings.warn(f"couldnt set model params - base_model/model_params inconsitent with scikit-learn")
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


def run_training_with_mlflow(mlflow_conf, 
                             sk_model,
                             train_dataloader, 
                             val_dataloader=None, 
                             test_dataloader=None,
                             **kwargs):
    tune = kwargs.get('tune', False)
    if tune:
        inner_cv = kwargs.get('inner_cv', C.DEFAULT_CV)
        h_search = kwargs.pop('h_search', None)
        if h_search is None:
            raise AttributeError(f'if tuner is requested, h_search should be provided')
        scoring = kwargs.get('scoring', C.DEFAULT_SCORING_CLASSIFIER)
        
    model_path = kwargs.pop('model_path', 'model')
    # model_save_dir = Path(kwargs.get('model_save_dir', C.MODEL_SAVE_DIR))
    # model_save_dir.mkdir(parents=True, exist_ok=True)
    artifacts = kwargs.pop('artifacts', dict())

    mlflow_conf.setdefault('problem_type', 'classifier')
    mlflow_setup = setup_mlflow(**mlflow_conf)

    calculate_metrics = Metrics(mlflow_conf['problem_type'])
    print(mlflow_setup, calculate_metrics)

    experiment_name = mlflow_setup['experiment_name']

    experiment_tags = dict()
    experiment_tags.update(**kwargs)

    with mlflow.start_run():
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

        # mlflow.log_params(sk_model.model.get_params())
        if tune:
            m, gs = sk_model.tune(X=_X, y=_y,
                                  hyper_params=h_search,
                                  cv=inner_cv, 
                                  experiment_name=experiment_name, 
                                  scoring=scoring)
            
            mlflow.sklearn.log_model(m, experiment_name + '_model')
            mlflow.sklearn.log_model(gs, experiment_name + '_GridSearchCV')
            
            log.info(f"Experiment: {experiment_name} has finished hyperparameter tuning")
            log.info("Hyperparameter search space: " + str(h_search))
            # log params
            mlflow.log_params(sk_model.params)
            print(f"Best_params:\n {gs.best_params_}")
        else:
            sk_model.fit(X=X_train, y=y_train)#, Xstd = X_train_std)
        
            mlflow.sklearn.log_model(sk_model.model, experiment_name)
            mlflow.log_params(sk_model.params)
            log.info(f"Experiment: {experiment_name} has finished training")

        for split_id, (train_index, val_index) in enumerate(outer_cv.split(_X, _y)):
            if split_id >= 1:
                warnings.warn("Current logic for tune and implicit outer_cv not correct")
                break

            _X_train, _X_val = _X[train_index, :], _X[val_index, :]
            _y_train, _y_val = _y[train_index], _y[val_index]
            
            y_val_proba = sk_model.predict_proba(_X_val)
            if y_val_proba.ndim > 1:
                y_val_proba = y_val_proba[:,1]

            y_val_hat = sk_model.predict(_X_val)
            val_score = sk_model.score(_X_val, _y_val)

        if test_dataloader is not None:
            y_test_proba = sk_model.predict_proba(X_test)
            if y_test_proba.ndim > 1:
                y_test_proba = y_test_proba[:, 1]
            y_test_hat = sk_model.predict(X_test)
            test_score = sk_model.score(X_test, y_test)
        else:
            y_test=None
            y_test_hat=None
            y_test_proba=None
            test_score =None

        # Calculate metrics
        sk_model.metrics = calculate_metrics(y_val=y_val, 
                                             y_val_proba=y_val_proba, 
                                             y_val_hat=y_val_hat,
                                             val_score=val_score, 
                                             y_test=y_test, 
                                             y_test_proba=y_test_proba, 
                                             y_test_hat=y_test_hat,
                                             test_score=test_score
                                            )
        _end_time = time.time()
        run_time = (_end_time - _start_time)
        
        # log metrics
        mlflow.log_metrics(sk_model.metrics)
        print(sk_model.metrics)

        experiment_tags.update(dict(run_time=run_time))
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)

        # Other artifacts
        _tmp = {f"artifact/{art_name}": art_val 
                for art_name, art_val in six.iteritems(artifacts)}
        helper.log_artifacts(_tmp, run_id, mlflow_uri=mlflow_setup['mlflow_uri'], delete=True) 

        return (run_id,
                sk_model.metrics,
                y_val, y_val_hat, y_val_proba,
                y_test, y_test_hat, y_test_proba,
                )
