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
from lightsaber.trainers.temperature_scaling import _ECELoss
from lightsaber.trainers.helper import setup_mlflow
from lightsaber.trainers.pt_regularizer import l1_regularization, l2_regularization
from lightsaber.trainers.components import BaseModel    # importing for backward compatibility

from functools import partial
import copy
from tempfile import NamedTemporaryFile
import warnings

import logging
log = logging.getLogger()


def load_model(model, ckpt_path):
    checkpoint = pl.utilities.cloud_io.load(ckpt_path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'on_load_checkpoint'): 
        model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model


class PyModel(pl.LightningModule):
    """PyModel"""
    def __init__(self, 
                 hparams:Namespace, 
                 model:nn.Module,
                 train_dataset: Optional[Dataset] = None, 
                 val_dataset: Optional[Dataset] = None,
                 cal_dataset: Optional[Dataset] = None, 
                 test_dataset: Optional[Dataset] = None,
                 collate_fn: Optional[Callable] = None, 
                 optimizer: Optional[Optimizer] = None,
                 loss_func: Optional[Callable] = None, 
                 out_transform: Optional[Callable] = None, 
                 num_workers: Optional[int] = 0, 
                 debug: Optional[bool] = False,
                 **kwargs):
        """
        Parameters
        ----------
        hparams: Namespace
            hyper-paramters for base model
        model: 
            base pytorch model defining the model logic. model forward should output logit for classfication and accept
            a single positional tensor (`x`) for input data and keyword tensors for `length` atleast. 
            Optinally can provide `hidden` keyword argument for sequential models to ingest past hidden state.
        train_dataset: torch.utils.data.Dataset, optional
            training dataset 
        val_dataset: torch.utils.data.Dataset, optional
            validation dataset 
        cal_dataset: torch.utils.data.Dataset, optional
            calibration dataset - if provided post-hoc calibration is performed
        test_dataset: torch.utils.data.Dataset, optional
            test dataset - if provided, training also report test peformance
        collate_fn: 
            collate functions to handle inequal sample sizes in batch
        optimizer: torch.optim.Optimizer, optional
            pytorch optimizer. If not provided, Adam is used with standard parameters
        loss_func: callable
            if provided, used to compute the loss. Default: cross entropy loss
        out_transform: callable
            if provided, convert logit to expected format. Default, softmax
        num_workers: int, Default: 0
            if provided sets the numer of workers used by the DataLoaders. 
        kwargs: dict, optional
            other parameters accepted by pl.LightningModule
        """
        super(PyModel, self).__init__()
        #  self.bk_hparams = hparams
        self.model = model

        self._debug = debug

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cal_dataset = cal_dataset
        self.test_dataset = test_dataset

        self.num_workers = num_workers

        self.collate_fn = collate_fn

        self._optimizer = optimizer
        self._scheduler = kwargs.get('scheduler', None)
        self._kwargs = kwargs

        # save hyper-parameters
        self.save_hyperparameters(hparams)

        # -------------------------------------------
        # TODO: Move to classifier
        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = loss_func

        if out_transform is None:
            self.out_transform = nn.Softmax(dim=1)
        else:
            self.out_transform = out_transform

        self.temperature = nn.Parameter(T.ones(1) * 1.)
        # -------------------------------------------
        return

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        if self._optimizer is None:

            optimizer = T.optim.Adam(self.model.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=1e-5  # standard value)
                                     )
        else:
            optimizer = self._optimizer
        
        if self._scheduler is None:
            return optimizer
        else:
            print("Here")
            return [optimizer], [self._scheduler]

    def on_load_checkpoint(self, checkpoint):
        # give sub model a chance to mess with the checkpoint
        if hasattr(self.model, 'on_load_checkpoint'):
            self.model.on_load_checkpoint(checkpoint)
        return

    # --------------------------------------------------------------
    #  Lightsaber:: 
    #  providing extra capabilities to model and compatibility with lightning 
    # -------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
                           
    def apply_regularization(self):
        """
        Applies regularizations on the model parameter
        """
        loss = 0.0
        if hasattr(self.hparams, 'l1_reg') and self.hparams.l1_reg > 0:
            loss += l1_regularization(self.parameters(), self.hparams.l1_reg)
        if hasattr(self.hparams, 'l2_reg') and self.hparams.l2_reg > 0:
            loss += l2_regularization(self.parameters(), self.hparams.l2_reg)
        return loss

    def freeze_except_last_layer(self):
        n_layers = sum([1 for _ in self.model.parameters()])
        freeze_layers = n_layers - 2
        i = 0
        freeze_number = 0
        free_number = 0
        for param in self.model.parameters():
            if i <= freeze_layers - 1:
                print('freezing %d-th layer' % i)
                param.requires_grad = False
                freeze_number += param.nelement()
            else:
                free_number += param.nelement()
            i += 1
        print('Total frozen parameters', freeze_number)
        print('Total free parameters', free_number)
        return 

    def clone(self):
        return copy.copy(self)
    
    # --------------------------------------------------------------
    #  Lightning:: step logic for train, test. validation
    # -------------------------------------------------------------
    def _common_step(self, batch, batch_idx):
        """Common step that is run over a batch of data. 

        Currently supports two types of data
        1. batch containing only X, y, corresponding lengths, and idx
        2. batch containing an extra dimension. Currently assuming its the summary data
        """
        # REQUIRED
        if len(batch) == 4:
            x, y, lengths, idx = batch
            y_out, _ = self.forward(x, lengths=lengths)
        elif len(batch) == 5:
            x, summary, y, lengths, idx = batch
            y_out, _ = self.forward(x, lengths=lengths, summary=summary)
        
        y_pred = self.out_transform(y_out)
        return (y_pred, y_out, y, x)
            
    def _shared_eval_step(self, y_pred, y_out, y, x, is_training=False):
        # Supporting loss functions that takes in X as well
        score = self._calculate_score(y_pred, y)
        n_examples = y.shape[0]

        is_x_included = False
        for param in signature(self.loss_func).parameters:
            if param == 'X':
                is_x_included = True
        
        if is_x_included:    
            loss = self.loss_func(y_out, y, X=x)
        else:
            loss = self.loss_func(y_out, y)
        
        if is_training:
            loss += (self.apply_regularization() / n_examples)
        # General way of classification
        return loss, n_examples, score

    # TODO: move this to classification
    def _process_common_output(self, y_pred):
        _, y_hat = T.max(y_pred.data, 1)
        return y_hat
    
    # TODO: make this an abstractmethod. currently done for classification
    def _calculate_score(self, y_pred, y):
        y_hat = self._process_common_output(y_pred)
        score = accuracy(y_hat, y)
        return score

    def training_step(self, batch, batch_idx):
        y_pred, y_out, y, x = self._common_step(batch, batch_idx)
        loss, n_examples, score = self._shared_eval_step(y_pred, y_out, y, x, is_training=True)
       
        # Making it independent of loggers used
        metrics = {"loss": loss, "train_score": score}
        self.log_dict(metrics, on_step=self._debug, on_epoch=True, prog_bar=True, logger=True) 
        if self._debug:
            self.log("train_n_examples", n_examples, on_step=True, on_epoch=True)
        #  tensorboard_log = {'batch_train_loss': loss, 'batch_train_score': train_score}
        return metrics  #, train_n_correct=n_correct, train_n_examples=n_examples, log=tensorboard_log)
    
    def validation_step(self, batch, batch_idx):
        y_pred, y_out, y, x = self._common_step(batch, batch_idx)
        loss, n_examples, score = self._shared_eval_step(y_pred, y_out, y, x)
       
        # Making it independent of loggers used
        metrics = {"val_loss": loss, "val_score": score}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        if self._debug:
            self.log("val_n_examples", n_examples, on_step=True, on_epoch=True)
        #  tensorboard_log = {'batch_val_loss': loss, 'batch_val_score': val_score}
        return metrics  #, val_n_correct=n_correct, val_n_examples=n_examples, log=tensorboard_log)

    def test_step(self, batch, batch_idx):
        y_pred, y_out, y, x = self._common_step(batch, batch_idx)
        loss, n_examples, score = self._shared_eval_step(y_pred, y_out, y, x)

        # Making it independent of loggers used
        metrics = {"test_loss": loss, "test_score": score}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        #  tensorboard_log = {'batch_test_loss': loss, 'batch_test_score': test_score}
        # For test returning both outputs and y
        #  y_pred = self._process_common_output(y_hat)
        #  metrics.update(dict(y_pred=y_pred, y_hat=y, y=y))
        return metrics #, test_n_correct=n_correct, test_n_examples=n_examples, log=tensorboard_log)

    def predict_step(self, batch, batch_idx):
        y_pred, y_out, y, x = self._common_step(batch, batch_idx)
        y_hat = self._process_common_output(y_pred)

        payload={'y_hat': y_hat, 'y_pred': y_pred, 'y': y}
        return payload

    def _on_predict_epoch_end(self, results):
        # TODO: this should be working directly as a model hook
        # Not working
        def _process_single_dataloader(res_dataloader):
            y_hat = T.cat([r['y_hat'] for r in res_dataloader])
            y_pred = T.cat([r['y_pred'] for r in res_dataloader])
            y = T.cat([r['y'] for r in res_dataloader])
            return dict(y_hat=y_hat, y_pred=y_pred, y=y)

        # making the code adaptive for multiple dataloaders
        log.debug(f"Number of predict dataloader: {len(self.trainer.predict_dataloaders)}")
        if len(self.trainer.predict_dataloaders) == 1:
            payload = _process_single_dataloader(results)
        else:
            payload = [_process_single_dataloader(res_dataloader) 
                       for res_dataloader in results]
        return payload
        
    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     try:
    #         avg_val_loss = T.stack([x['batch_val_loss'] for x in outputs]).mean()
    #     except Exception:
    #         avg_val_loss = T.FloatTensor([0.])
    #     
    #     try:
    #         val_score = (np.stack([x['val_n_correct'] for x in outputs]).sum() 
    #                      / np.stack([x['val_n_examples'] for x in outputs]).sum())
    #     except Exception:
    #         val_score = T.FloatTensor([0.])
    #         
    #     tensorboard_log = {'val_loss': avg_val_loss, 'val_score': val_score}
    #     return dict(val_loss=avg_val_loss, val_score=val_score, log=tensorboard_log)

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
    def get_params(self):
        """Return a dicitonary of param_name: param_value
        """
        _params = vars(self.hparams)
        return _params

    # TODO: Move to classifier
    def predict_proba(self, *args, **kwargs):
        logit, _ = self.forward(*args, **kwargs)
        pred = self.out_transform(self.temperature_scale(logit))
        return pred

    # TODO: Move to classifier
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

    # --------------------------------------------------------------
    #  Dataset handling section
    # TODO: move to dataset class
    # -------------------------------------------------------------
    def _pin_memory(self):
        pin_memory = False
        try:
            if self.trainer.on_gpu:
                pin_memory=True
        except AttributeError:
            pass
        return pin_memory

    def train_dataloader(self):
        warnings.warn(f'{C._deprecation_warn_msg}. Pass dataloader directly', DeprecationWarning, stacklevel=2)
        sampler = self._kwargs.get('train_sampler', None)
        shuffle = True if sampler is None else False

        pin_memory = self._pin_memory()
        # REQUIRED
        dataloader = DataLoader(self.train_dataset, 
                                collate_fn=self.collate_fn, 
                                shuffle=shuffle,
                                batch_size=self.hparams.batch_size,
                                sampler=sampler,
                                pin_memory=pin_memory,
                                num_workers=self.num_workers
                                )
        return dataloader

    def val_dataloader(self):
        warnings.warn(f'{C._deprecation_warn_msg}. Pass dataloader directly', DeprecationWarning, stacklevel=2)
        if self.val_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.val_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.val_dataset, 
                                    collate_fn=self.collate_fn, 
                                    pin_memory=pin_memory,
                                    batch_size=self.hparams.batch_size,
                                    num_workers=self.num_workers
                                    )
        return dataloader

    def test_dataloader(self):
        warnings.warn(f'{C._deprecation_warn_msg}. Pass dataloader directly', DeprecationWarning, stacklevel=2)
        if self.test_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.test_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.test_dataset, 
                                    collate_fn=self.collate_fn,
                                    pin_memory=pin_memory,
                                    batch_size=self.hparams.batch_size,
                                    num_workers=self.num_workers)
        return dataloader

    def cal_dataloader(self):
        warnings.warn(f'{C._deprecation_warn_msg}. Pass dataloader directly', DeprecationWarning, stacklevel=2)
        if self.cal_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.cal_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.cal_dataset, collate_fn=self.collate_fn, pin_memory=pin_memory,
                                    batch_size=self.hparams.batch_size,num_workers=self.num_workers)
        return dataloader
    

# TODO: see if this is still required
def _find_checkpoint(model, checkpoint_path=None):
    if checkpoint_path is None:
        if model is not None and hasattr(model, 'trainer'):
            trainer = model.trainer

            # do nothing if there's not dir or callback
            no_ckpt_callback = (trainer.checkpoint_callback is None) or (not trainer.checkpoint_callback)
            if no_ckpt_callback or not os.path.exists(trainer.checkpoint_callback.filepath):
                pass
            else:
                # restore trainer state and model if there is a weight for this experiment
                last_epoch = -1
                last_ckpt_name = None

                # find last epoch
                checkpoints = os.listdir(trainer.checkpoint_callback.filepath)
                for name in checkpoints:
                    # ignore hpc ckpts
                    if 'hpc_' in name:
                        continue

                    if '.ckpt' in name:
                        epoch = name.split('epoch_')[1]
                        epoch = int(re.sub('[^0-9]', '', epoch))

                        if epoch > last_epoch:
                            last_epoch = epoch
                            last_ckpt_name = name
                if last_ckpt_name is not None:
                    checkpoint_path = os.path.join(trainer.checkpoint_callback.filepath, last_ckpt_name)
    return checkpoint_path


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


def run_training_with_mlflow(mlflow_conf: dict,
                             train_args: Namespace,
                             wrapped_model: PyModel,
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


def load_model_from_mlflow(run_id, 
                           mlflow_conf,
                           wrapped_model,
                           model_path="model_checkpoint",
                           ):
    """Method to load a trained model from mlflow

    Parameters
    ----------
    run_id: str
        mlflow run id for the trained model
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    wrapped_model: PyModel
        model architecture to be logged
    model_path: str
        output path where model checkpoints are logged

    Returns
    -------
    PyModel:
        wrapped model with saved weights and parameters from the run
    """
    mlflow_setup = helper.setup_mlflow(**mlflow_conf)
    #  model_uri = f"runs:/{run_id}/{mlflow_setup['experiment_name']}_{model_path}"
    # run_data = helper.fetch_mlflow_run(run_id, 
    #                                    mlflow_uri=mlflow_setup['mlflow_uri'],
    #                                    parse_params=True
    #                                    )

    # hparams = run_data['params']
    # model_name = run_data['tags']['model']
    # if wrapped_model is None:
    #     base_model = mlflow.sklearn.load_model(model_uri)
    #     wrapped_model = SKModel(base_model, hparams, name=model_name)

    run_data = helper.fetch_mlflow_run(run_id, 
                                       mlflow_uri=mlflow_setup['mlflow_uri'],
                                       artifacts_prefix=[model_path])

    ckpt_path = helper.get_artifact_path(run_data['artifact_paths'][0], 
                                         artifact_uri=run_data['info'].artifact_uri)
    wrapped_model = load_model(wrapped_model, ckpt_path)
    return wrapped_model


def register_model_with_mlflow(run_id, 
                               mlflow_conf,
                               wrapped_model,
                               registered_model_name,
                               model_path='model_checkpoint',
                               **artifacts
                               ):
    """Method to register a trained model

    Parameters
    ----------
    run_id: str
        mlflow run id for the trained model
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    wrapped_model: PyModel
        model architecture to be logged
    registered_model_name: str
        name for registering the model
    model_path: str
        output path where model will be logged
    artifacts: dict
        dictionary of objects to log with the model
    """
    # Getting run info
    mlflow_setup = helper.setup_mlflow(**mlflow_conf)
    wrapped_model = load_model_from_mlflow(run_id, mlflow_conf, wrapped_model, model_path)
    # Registering model
    try:
        mlflow.pytorch.log_model(wrapped_model, model_path.rstrip('_checkpoint'), registered_model_name=registered_model_name)
    except Exception as e:
        log.error(f'Exception during logging model: {e}. Continuing to dump artifacts')

    # logging other artifacts
    dumper = helper.model_register_dumper(registered_model_name=registered_model_name)
    helper.log_artifacts(artifacts, run_id, mlflow_uri=mlflow_setup['mlflow_uri'], dumper=dumper, delete=True)
    return 
