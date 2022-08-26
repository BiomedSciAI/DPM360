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
# limitations under the License.

from abc import abstractmethod, ABC
import os
import json
import pickle
import socket
import numpy as np
import tqdm
import re
import time
import six
from inspect import signature

from typing import Optional, Callable

import torch as T
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from argparse import ArgumentParser, Namespace

# For num_workers >1 in Dataloader
# try:
#     T.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch

from torchmetrics.functional import accuracy

from lightsaber import constants as C
from lightsaber.data_utils import utils as du
from lightsaber.data_utils import pt_dataset as ptd
from lightsaber.metrics import Metrics
from lightsaber.trainers import helper
from lightsaber.trainers.helper import setup_mlflow
from lightsaber.trainers.components import _ECELoss
from lightsaber.trainers.components import BaseTask, load_model, _find_checkpoint
from lightsaber.trainers.components import BaseModel    # importing for backward compatibility

from functools import partial
import copy
from tempfile import NamedTemporaryFile
import warnings

import logging
log = logging.getLogger()


class ClassificationTask(BaseTask):
    """classification Task"""

    # --------------------------------------------------------------
    #   Implementing abstract methods
    # -------------------------------------------------------------
    def _setup_task(self):
        if self._loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = self._loss_func

        if self._out_transform is None:
            self.out_transform = nn.Softmax(dim=1)
        else:
            self.out_transform = self._out_transform

        self.temperature = nn.Parameter(T.ones(1) * 1.)
        # -------------------------------------------
        return

    def _process_common_output(self, y_pred):
        _, y_hat = T.max(y_pred.data, 1)
        return y_hat
    
    def _calculate_score(self, y_pred, y):
        y_hat = self._process_common_output(y_pred)
        score = accuracy(y_hat, y)
        return score

    # --------------------------------------------------------------
    #   Classifier specific section:: calibration
    # -------------------------------------------------------------
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, cal_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        _orig_device = self.device
        try:
            if self.trainer.on_gpu:
                self.to(self.trainer.root_gpu)
        except Exception:
            pass
        #  self.cuda()
        self.temperature.data = T.ones(1, device=self.temperature.device) * 1.5

        # nll_criterion = nn.CrossEntropyLoss()
        nll_criterion = self.loss_func
        ece_criterion = _ECELoss()
        n_batches = len(cal_loader)
            
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with T.no_grad():
            # making it compatible with non trainer run
            try:
                if self.trainer.on_gpu:
                    nll_criterion = self.trainer.transfer_to_gpu(nll_criterion, self.trainer.root_gpu)
                    ece_criterion = self.trainer.transfer_to_gpu(ece_criterion, self.trainer.root_gpu)
            except Exception:
                pass

            for (bIdx, batch) in tqdm.tqdm(enumerate(cal_loader), total=n_batches):
                if bIdx == n_batches:
                    break
                
                # making it compatible with non trainer run
                try:       
                    if self.trainer.on_gpu:
                        batch = self.trainer.transfer_batch_to_gpu(batch, self.trainer.root_gpu)
                except Exception:
                    pass
            #  for input, label in cal_loader:
                if len(batch) == 4:
                    x, y, lengths, idx = batch 
                    logits, _ = self.forward(x, lengths)
                elif len(batch) == 5:
                    x, summary, y, lengths, idx = batch
                    logits, _ = self.forward(x, lengths, summary)
                logits_list.append(logits)
                labels_list.append(y)
            logits = T.cat(logits_list)
            labels = T.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        self.to(_orig_device)
        return self

    # --------------------------------------------------------------
    #  Scikit-learn compatibility section
    # -------------------------------------------------------------
    def predict_proba(self, *args, **kwargs):
        logit, _ = self.forward(*args, **kwargs)
        pred = self.out_transform(self.temperature_scale(logit))
        return pred

    def predict(self, *args, **kwargs):
        proba = self.predict_proba(*args, **kwargs)
        pred = T.argmax(proba, dim=-1)
        return pred

    # DPM360:: connector
    # Given the patient id, find the array index of the patient
    def predict_patient(self, patient_id, test_dataset):
        p_x, _, p_lengths, _ = test_dataset.get_patient(patient_id)
        proba = self.predict_proba(p_x, lengths=p_lengths)
        return proba
    

# TODO: see if this can be broken down and reusable
# potentially move to to a mlflow section
def post_training(trainer, wrapped_model, ckpt_path, model_path, **kwargs):
    cal_dataloader = wrapped_model.cal_dataloader()
    if len(cal_dataloader) > 0 and kwargs.get('calibrate', False):
        wrapped_model.set_temperature(cal_dataloader)
        # manually setting the best model and the model mode
        if ckpt_path is not None:
            ckpt_path = os.path.join(os.path.dirname(ckpt_path),
                                     f"{os.path.basename(ckpt_path).rstrip('.ckpt')}-calibrated.ckpt"
                                     )
        else:
            ckpt_path = os.path.join(os.getcwd(), f"{model_path}-calibrated.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Calibrated model saved to {ckpt_path}")
    return wrapped_model, ckpt_path


def run_classification_with_mlflow(mlflow_conf: dict,
                                   train_args: Namespace,
                                   wrapped_model: ClassificationTask,
                                   train_dataloader=None,
                                   val_dataloader=None,
                                   test_dataloader=None,
                                   cal_dataloader=None,
                                   **kwargs):
    """
    Function to run supervised training for classifcation

    Parameters
    ----------
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    train_args: Namespace
        namespace with arguments for pl.Trainer instance. See `pytorch_lightning.trainer.Trainer` for supported options
        TODO: potentially hyper-parameters for model
    wrapped_model: PyModel
        wrapped PyModel
    train_dataloader: DataLoader, optional
        training dataloader
        If not provided dataloader is extracted from `wrapped_model` (backwards compatibility)
    val_dataloader: DataLoader, optional
        validation dataloader.
        If not provided dataloader is extracted from `wrapped_model` (backwards compatibility)
    test_dataloader: DataLoader, optional
        test dataloader
        If not provided dataloader is extracted from `wrapped_model` (backwards compatibility)
    cal_dataloader: DataLoader, optional
        calibration dataloader
        If not provided dataloader is extracted from `wrapped_model` (backwards compatibility)
    model_path: str
        prefix for storing model in MlFlow
    artifacts: dict, optional
        any artifact to be logged by user
    metrics: Callable, optional
        if specified, used for calculating all metrics. else inferred from problem type
    run_id: str, optional
        if specified uses existing mlflow run.
    auto_init_logger: bool, default: True
        if specificed, loggers are generated automatically.
        else, assumes user passed it (this is planned and not implemented now)
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

    # **Collecting the dataloaders**
    # TODO: deprecate this in next version
    if train_dataloader is None:
        train_dataloader = wrapped_model.train_dataloader()
    if val_dataloader is None:
        val_dataloader = wrapped_model.val_dataloader()
    if test_dataloader is None:
        test_dataloader = wrapped_model.test_dataloader()
    if cal_dataloader is None:
        cal_dataloader = wrapped_model.cal_dataloader()

    auto_init_logger = kwargs.pop('auto_init_logger', True)
    if auto_init_logger:
        mlf_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_name,
                                             tracking_uri=mlflow_setup['mlflow_uri'],
                                             tags=experiment_tags
                                             )
        _logger = [mlf_logger]
    else:
        # Possible logic:
        # 1. parse train_args to strip the logger if exists
        # 2. assert MlFlowLogger is one of the logger present
        mlf_logger = None
        _logger = []
        raise NotImplementedError('TODO: allow users to specify logger')

    # Trying to resume a run
    # N.B. We are monkey patching this here so that future user supplied logger
    #      can be modified to correct run_id
    run_id = kwargs.pop('run_id', None)
    if run_id is not None:
        mlf_logger._run_id = run_id

    _parsed_train_args = pl.Trainer.parse_argparser(train_args)
    trainer = pl.Trainer.from_argparse_args(_parsed_train_args, logger=_logger)

    # Starting to train
    _start_time = time.time()
    trainer.fit(wrapped_model, 
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                )

    run_id = mlf_logger._run_id  #TODO: when supporting user supplied logger, check this to get the correct mlflow run id
    #  mlflow.log_params(wrapped_model.get_params())

    # Finding the checkpoint path
    try:
        ckpt_path = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                ckpt_path = callback.best_model_path
                break
        if ckpt_path is None:
            raise Exception('couldnt determine the best model from callbacks..falling back to file matching')
    except Exception:
        ckpt_path = _find_checkpoint(wrapped_model)
    log.info(f"Best model is temporarily in {ckpt_path}")

    try:
        checkpoint = pl.utilities.cloud_io.load(ckpt_path)['state_dict']
        wrapped_model.load_state_dict(checkpoint)
    except Exception as e:
        raise Exception(f"couldnt restore model properly from {ckpt_path}. Error={e}")

    # Post training routines. e.g. Calibrating if calibration requested
    wrapped_model, ckpt_path = post_training(trainer, wrapped_model, ckpt_path, model_path)
    # Setting up for evaluation loop
    wrapped_model.eval()

    # Collecting metrics
    if (not isinstance(val_dataloader, DataLoader)) and (len(val_dataloader) > 1):
        raise NotImplementedError("For now supporting only one val dataloader")
    val_payload = trainer.predict(wrapped_model, val_dataloader)
    # TODO: remove this step if the bug is fixed
    val_payload = wrapped_model._on_predict_epoch_end(val_payload)
                                               
    y_val = val_payload['y'].data.cpu().numpy()
    y_val_pred = val_payload['y_pred'].data.cpu().numpy()
    y_val_hat = val_payload['y_hat'].data.cpu().numpy()

    if len(test_dataloader) > 0:
        if (not isinstance(test_dataloader, DataLoader)) and (len(test_dataloader) > 1):
            raise NotImplementedError("For now supporting only one test dataloader")
        test_payload = trainer.predict(wrapped_model, test_dataloader)
        # TODO: remove this step if the bug is fixed
        test_payload = wrapped_model._on_predict_epoch_end(test_payload)

        y_test = test_payload['y'].data.cpu().numpy()
        y_test_pred = test_payload['y_pred'].data.cpu().numpy()
        y_test_hat = test_payload['y_hat'].data.cpu().numpy()
    else:
        y_test, y_test_pred, y_test_hat = None, None, None 

    _end_time = time.time()
    run_time = (_end_time - _start_time)
    # Experiment ended. logging things
    experiment_tags.update(dict(run_time=run_time))

    try:
        run_metrics = calculate_metrics(y_val=y_val, y_val_hat=y_val_hat, y_val_proba=y_val_pred, 
                                        y_test=y_test, y_test_hat=y_test_hat, y_test_proba=y_test_pred)

    except Exception as e:
        warnings.warn(f"{e}")
        log.warning(f"something went wrong while computing metrics: {e}")
        run_metrics = None

    helper.log_metrics(run_metrics, run_id=run_id)

    _tmp = {f"artifact/{art_name}": art_val 
            for art_name, art_val in six.iteritems(artifacts)}
    _tmp['model_checkpoint'] = ckpt_path
    helper.log_artifacts(_tmp, run_id=run_id, mlflow_uri=mlflow_setup['mlflow_uri'], delete=True) 

    helper.set_tags(experiment_tags, run_id=run_id)

    # Pytorch log model not working
    # *****************************
    #  mlflow.pytorch.log_model(wrapped_model, model_path, registered_model_name=problem_type)     # <------ use mlflow.pytorch.log_model to log trained sklearn model
    #  print("Model saved in run {}, and registered on {} as a new version of model name {}"
    #       .format(active_run, os.environ['MLFLOW_URI'], problem_type))

    return (run_id, 
            run_metrics, 
            y_val, y_val_hat, y_val_pred,
            y_test, y_test_hat, y_test_pred,
            )
