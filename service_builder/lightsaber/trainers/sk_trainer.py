#!/usr/bin/env python
# Author : James Codella<jvcodell@us.ibm.com>, Prithwish Chakraborty <prithwish.chakraborty@ibm.com>
# date   : 2020-06-02

import numpy as np
from pathlib import Path

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.experimental import enable_hist_gradient_boosting

import mlflow

from lightsaber import constants as C
from lightsaber.metrics import calculate_metrics
from lightsaber.data_utils import sk_dataloader as skd
from lightsaber.data_utils import utils as du
from lightsaber.trainers.helper import (setup_mlflow, get_model, get_predefined_split,
                                        load_sk_model, save_sk_model, import_model_class)


# ***********************************************************************
#         SK Model Trainer
# ***********************************************************************
class SKModel(object):
    def __init__(self, 
                 base_model,
                 model_params=None, 
                 name = "undefined_model_name"):
        super(SKModel, self).__init__()
        self.model = base_model
        self.model_params = model_params
        self.__name__ = name
        
        self.metrics = {}
        self.proba = []
        # self.params = self.model.get_params()
 
    @property
    def params(self):
        try:
            params = self.model.get_params()
        except AttributeError:
            params = self.model_params
        return params

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
        self.params = self.model.get_params()
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

    def tune(self, X, y,
             hyper_params,
             experiment_name, 
             cv=C.DEFAULT_CV,
             scoring='roc_auc',
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
        self.params = gs.best_params_
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


def model_init(hparams):
    expt_conf = du.ConfReader(hparams.config)

    idx_col = expt_conf['idx_col']
    tgt_col = expt_conf['tgt_col']
    feat_cols = expt_conf['feat_cols'] 
    category_map = expt_conf.get('category_map', C.DEFAULT_MAP)

    _data_conf = expt_conf.get('data', dict())
    fill_value = _data_conf.get('fill_value', 0.)
    preprocessor = _data_conf.get('preprocessor', None)
    if preprocessor is not None:
        for idx, p in preprocessor:
            preprocessor[idx] = import_model_class(p)

    flatten = _data_conf.get('flatten', C.DEFAULT_FLATTEN)

    train_dataloader = skd.SKDataLoader(tgt_file=expt_conf['train']['tgt_file'],
                                        feat_file=expt_conf['train']['feat_file'],
                                        idx_col=idx_col,
                                        tgt_col=tgt_col,
                                        feat_columns=feat_cols,
                                        category_map=category_map,
                                        fill_value=fill_value,
                                        flatten=flatten,
                                        preprocessor=preprocessor)
    fitted_preprocessor = train_dataloader.get_preprocessor(refit=True)

    if 'val' in expt_conf:
        val_dataloader = skd.SKDataLoader(tgt_file=expt_conf['val']['tgt_file'],
                                          feat_file=expt_conf['val']['feat_file'],
                                          idx_col=idx_col,
                                          tgt_col=tgt_col,
                                          feat_columns=feat_cols,
                                          category_map=category_map,
                                          fill_value=fill_value,
                                          flatten=flatten,
                                          preprocessor=fitted_preprocessor)
    else:
        val_dataloader = None
        
    print("Datasets ready")
    payload = dict(train_dataloader=train_dataloader, 
                   val_dataloader=val_dataloader)

    sk_model_type = expt_conf['model']['type']
    if hparams.load:
        load_path = expt_conf['model']['model_path']
        sk_model = load_sk_model(load_path)
    else:
        sk_model_parameters = expt_conf['model'].get('params', None)
        sk_model = get_model(sk_model_type, sk_model_parameters)

    if hparams.tune:
        h_search = expt_conf['model']['hyperparams_search']
        payload.update(dict(h_search=h_search))
    print("*** Done processing data...")
    return (sk_model, hparams.conf, payload)


def run_training_with_mlflow(mlflow_conf, 
                             sk_model,
                             train_dataloader, 
                             val_dataloader=None, 
                             test_dataloader=None,
                             **kwargs):
    #  mlflow_conf = conf.get('mlflow', dict())
    mlflow_setup = setup_mlflow(**mlflow_conf)
    tune = kwargs.get('tune', False)

    model_save_dir = Path(kwargs.get('model_save_dir', C.MODEL_SAVE_DIR))
    model_save_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train_dataloader.read_data()
    
    if val_dataloader is not None:
        X_val, y_val = val_dataloader.read_data()
        cv, _X, _y = get_predefined_split(X_train, y_train, X_valid, y_valid)
    else:
        cv = kwargs.get('cv', C.DEFAULT_CV)
        _X = X_train
        _y = y_train

    if test_dataloader is not None:
        X_test, y_test = test_dataloader.read_data()

    experiment_name = mlflow_setup['experiment_name']

    print(mlflow_setup)

    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model.model, experiment_name)
        mlflow.log_params(sk_model.model.get_params())

        if tune:
            h_search = kwargs['h_search']
            scoring = kwargs['scoring']
            m, gs = sk_model.tune(X=_X, y=_y,
                                  hyper_params=h_search,
                                  cv=cv, 
                                  experiment_name=experiment_name, 
                                  scoring=scoring)
            
            mlflow.sklearn.log_model(m, experiment_name + '_model')
            mlflow.sklearn.log_model(gs, experiment_name + '_GridSearchCV')
            
            print("*** Experiment: " + experiment_name + " has finished hyperparameter tuning. ***")
            print("Hyperparameter search space: " + str(h_search))
            # log params
            mlflow.log_params(sk_model.params)
            print("Best_params:")
            print(gs.best_params_)
            save_sk_model(sk_model, model_save_dir + "/" + experiment_name)
        else:
            sk_model.fit(X=X_train,y=y_train)#, Xstd = X_train_std)
        
            mlflow.sklearn.log_model(sk_model.model, experiment_name)
            mlflow.log_params(sk_model.params)
        
            print("*** Experiment: " + experiment_name + " has finished training. ***")
            save_sk_model(sk_model, model_save_dir + '/' + 'experiment_name')

        for split_id, (train_index, val_index) in enumerate(cv):
            _X_train, _X_val = _X[train_index,:], X[test_index,:]
            _y_train, _y_val = _y[train_index], _y[test_index]
            
            y_val_proba = sk_model.predict_proba(X_val)
            if y_val_proba.ndim > 1:
                y_val_proba = y_val_proba[:,1]

            y_val_hat = sk_model.predict(X_val)
            val_score = sk_model.score(X_val, y_val)

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
        # log metrics
        mlflow.log_metrics(sk_model.metrics)
        print(sk_model.metrics)

        payload = dict(mode=sk_model,
                       y_val=y_val, 
                       y_val_hat=y_val_hat,
                       y_val_proba=y_val_proba,
                       y_test=y_test, 
                       y_test_hat=y_test_hat,
                       y_test_proba=y_test_proba,
                      )
        return payload
