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

import torch as T
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# For num_workers >1 in Dataloader
# try:
#     T.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch

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
    def __init__(self, hparams, model,
                 train_dataset, val_dataset,
                 cal_dataset=None, test_dataset=None,
                 collate_fn=None, optimizer=None,
                 loss_func=None, out_transform=None, num_workers=0, **kwargs):
        """
        Parameters
        ----------
        hparams: Namespace
            hyper-paramters for base model
        model: 
            base pytorch model defining the model logic. model forward should output logit for classfication and accept
            a single positional tensor (`x`) for input data and keyword tensors for `length` atleast. 
            Optinally can provide `hidden` keyword argument for sequential models to ingest past hidden state.
        train_dataset: torch.utils.data.Dataset
            training dataset 
        val_dataset: torch.utils.data.Dataset
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
        self.bk_hparams = hparams
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cal_dataset = cal_dataset
        self.test_dataset = test_dataset

        self.num_workers = num_workers

        self.collate_fn = collate_fn

        self._optimizer = optimizer
        self._scheduler = kwargs.get('scheduler', None)
        self._kwargs = kwargs

        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = loss_func

        if out_transform is None:
            self.out_transform = nn.Softmax(dim=1)
        else:
            self.out_transform = out_transform

        self.temperature = nn.Parameter(T.ones(1) * 1.)
        return

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

    def on_load_checkpoint(self, checkpoint):
        # give sub model a chance to mess with the checkpoint
        if hasattr(self.model, 'on_load_checkpoint'):
            self.model.on_load_checkpoint(checkpoint)
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
                           
    def predict_proba(self, *args, **kwargs):
        logit, _ = self.forward(*args, **kwargs)
        pred = self.out_transform(self.temperature_scale(logit))
        return pred

    def predict(self, *args, **kwargs):
        proba = self.predict_proba(*args, **kwargs)
        pred = T.argmax(proba, dim=-1)
        return pred

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        if self._optimizer is None:
            optimizer = T.optim.Adam(self.model.parameters(),
                                     lr=self.bk_hparams.lr,
                                     weight_decay=1e-5  # standard value)
                                    )
        else:
            optimizer = self._optimizer
        
        if self._scheduler is None:
            return optimizer
        else:
            print("Here")
            return [optimizer], [self._scheduler]
    
    def _common_step(self, batch, batch_idx):
        # REQUIRED
        if len(batch) == 4:
            x, y, lengths, idx = batch
            y_logits, _ = self.forward(x, lengths=lengths)
        elif len(batch) == 5:
            x, summary, y, lengths, idx = batch
            y_logits, _ = self.forward(x, lengths=lengths, summary=summary)
            
        X_included = False
        for param in signature(self.loss_func).parameters:
            if param == 'X':
                X_included = True
        if X_included:    
            loss = self.loss_func(y_logits, y, X = x)
        else:
            loss = self.loss_func(y_logits, y)
        
        outputs = self.out_transform(y_logits)
        _, predicted = T.max(outputs.data, 1)
        
        n_correct = (predicted == y).sum().item()
        n_examples = y.size(0)
        return loss, n_correct, n_examples

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_examples = self._common_step(batch, batch_idx)
        loss += (self.apply_regularization() / n_examples)
        train_score = n_correct / n_examples
        tensorboard_log = {'batch_train_loss': loss, 'batch_train_score': train_score}
        return dict(loss=loss, train_score=train_score, log=tensorboard_log) #, train_n_correct=n_correct, train_n_examples=n_examples, log=tensorboard_log)

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        loss, n_correct, n_examples = self._common_step(batch, batch_idx)
        return dict(batch_val_loss=loss, val_n_correct=n_correct, val_n_examples=n_examples)

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        loss, n_correct, n_examples = self._common_step(batch, batch_idx)
        return dict(batch_test_loss=loss, teest_n_correct=n_correct, test_n_examples=n_examples)

    def validation_end(self, outputs):
        # OPTIONAL
        try:
            avg_val_loss = T.stack([x['batch_val_loss'] for x in outputs]).mean()
        except Exception:
            avg_val_loss = T.FloatTensor([0.])
        
        try:
            val_score = (np.stack([x['val_n_correct'] for x in outputs]).sum() 
                         / np.stack([x['val_n_examples'] for x in outputs]).sum())
        except Exception:
            val_score = T.FloatTensor([0.])
            
        tensorboard_log = {'val_loss': avg_val_loss, 'val_score': val_score}
        return dict(val_loss=avg_val_loss, val_score=val_score, log=tensorboard_log)

    def _pin_memory(self):
        pin_memory = False
        try:
            if self.trainer.on_gpu:
                pin_memory=True
        except AttributeError:
            pass
        return pin_memory

    
    def train_dataloader(self):
        sampler = self._kwargs.get('train_sampler', None)
        shuffle = True if sampler is None else False

        pin_memory = self._pin_memory()
        # REQUIRED
        dataloader = DataLoader(self.train_dataset, collate_fn=self.collate_fn, shuffle=shuffle,
                                batch_size=self.bk_hparams.batch_size,
                                sampler=sampler, pin_memory=pin_memory,
                                num_workers=self.num_workers)
        return dataloader

    
    def val_dataloader(self):
        if self.val_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.val_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.val_dataset, collate_fn=self.collate_fn, pin_memory=pin_memory,
                                    batch_size=self.bk_hparams.batch_size,num_workers=self.num_workers)
        return dataloader

 
    def test_dataloader(self):
        if self.test_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.test_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.test_dataset, collate_fn=self.collate_fn, pin_memory=pin_memory,
                                    batch_size=self.bk_hparams.batch_size,num_workers=self.num_workers)
        return dataloader

    
    def cal_dataloader(self):
        if self.cal_dataset is None:
            dataset = ptd.EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.cal_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.cal_dataset, collate_fn=self.collate_fn, pin_memory=pin_memory,
                                    batch_size=self.bk_hparams.batch_size,num_workers=self.num_workers)
        return dataloader
    
    def get_params(self):
        """Return a dicitonary of param_name: param_value
        """
        _params = vars(self.bk_hparams)
        return _params
    
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

    def run(self, dataloader, overfit_pct=0):        
        _orig_device = self.device
        try:
            if self.trainer.on_gpu:
                self.to(self.trainer.root_gpu)
        except Exception:
            pass

        n_batches = len(dataloader)
        if overfit_pct > 0:
            n_batches = int(n_batches * overfit_pct)
            
        all_preds = []
        all_y = []
        
        for (bIdx, batch) in tqdm.tqdm(enumerate(dataloader), total=n_batches):
            if bIdx == n_batches:
                break
            
            # making it compatible with non trainer run
            try:       
                if self.trainer.on_gpu:
                    batch = self.trainer.transfer_batch_to_gpu(batch, self.trainer.root_gpu)
            except Exception:
                pass
                
            if len(batch) == 4:
                x, y, lengths, idx = batch 
                pred = self.predict_proba(x, lengths)
            elif len(batch) == 5:
                x, summary, y, lengths, idx = batch
           
                # x.to(device)
                # summary.to(device)
                # y.to(device)
                
                pred = self.predict_proba(x, lengths, summary)

            all_preds.append(pred)
            all_y.append(y)
        
        preds = T.cat(all_preds, dim=0)
        y = T.cat(all_y, dim=0)
        
        _, yhat = T.max(preds, dim=1)
        
        _n_correct = (yhat == y).sum().item()
        _n_examples = y.size(0)
        score = (_n_correct / _n_examples)

        self.to(_orig_device)
        return preds, yhat, y, score
    
    def freeze_except_last_layer(self):
        n_layers = sum([1 for _ in self.model.parameters()])
        freeze_layers = n_layers - 2
        i = 0
        freeze_number = 0
        free_number = 0
        for param in self.model.parameters():
            if i <= freeze_layers - 1:
                print ('freezing %d-th layer' % i)
                param.requires_grad = False
                freeze_number += param.nelement()
            else:
                free_number += param.nelement()
            i += 1
        print ('Total frozen parameters', freeze_number)
        print ('Total free parameters', free_number)
        return 

    def clone(self):
        return copy.copy(self)

    # Given the patient id, find the array index of the patient
    def predict_patient(self, patient_id, test_dataset):
        p_x, _, p_lengths, _ = test_dataset.get_patient(patient_id)
        proba = self.predict_proba(p_x, lengths=p_lengths)
        return proba


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


def run_training_with_mlflow(mlflow_conf, trainer, wrapped_model, **kwargs):
    """
    Function to run supervised training for classifcation

    Parameters
    ----------
    mlflow_conf: dict
        mlflow configuration e,g, MLFLOW_URI
    trainer: pl.Trainer
        a pytorch lightning trainer implementing `fit` function
    wrapped_model: PyModel
        wrapped PyModel 
    kwargs: dict of dicts, optional
        can contain `artifacts` to log with models, `model_path` to specify model output path, and remianing used as experiment tags
        
    Returns
    -------
    tuple:
        (run_id, run_metrics, val_y, val_yhat, val_pred_proba, test_y, test_yhat, test_pred_proba)
    """
    model_path = kwargs.pop('model_path', 'model')
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
        trainer.fit(wrapped_model)

        mlflow.log_metric('train_score', trainer.callback_metrics['train_score'])
        mlflow.log_metric('train_loss', float(trainer.callback_metrics['loss'].data.cpu().numpy()))
        mlflow.log_metric('val_loss', float(trainer.callback_metrics['val_loss'].data.cpu().numpy()))
        mlflow.log_params(wrapped_model.get_params())
        
        try:
            ckpt_path = None
            for callback in trainer.callbacks:
                if isinstance(callback, pl.callbacks.ModelCheckpoint):
                    ckpt_path = callback.best_model_path
                    break
            if ckpt_path is None:
                raise Exception('couldnt determine the best model')
        except Exception as e:
            ckpt_path = _find_checkpoint(wrapped_model)
        print(f"Best model is temporarily in {ckpt_path}")

        try:
            checkpoint = pl.utilities.cloud_io.load(ckpt_path)['state_dict']
            wrapped_model.load_state_dict(checkpoint)
        except Exception as e:
            raise Exception(f"couldnt restore model properly from {ckpt_path}. Error={e}")

        # Calibrating if calibration requested
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

        wrapped_model.eval()

        # Collecting metrics
        val_dataloader = wrapped_model.val_dataloader()
        (val_pred_proba, val_yhat, 
         val_y, val_score) = wrapped_model.run(val_dataloader, 
                                                overfit_pct=kwargs.get('overfit_pct', 0))
        y_val = val_y.data.cpu().numpy()
        y_val_proba = val_pred_proba[:, 1].data.cpu().numpy()
        y_val_hat = val_yhat.data.cpu().numpy()

        test_dataloader = wrapped_model.test_dataloader()
        if len(test_dataloader) > 0:
            log.warning("For now supporting only one test dataloader")

            (test_pred_proba, test_yhat, 
             test_y, test_score) = wrapped_model.run(test_dataloader, 
                                                     overfit_pct=kwargs.get('overfit_pct', 0))
            y_test = test_y.data.cpu().numpy()
            y_test_proba = test_pred_proba[:, 1].data.cpu().numpy()
            y_test_hat = test_yhat.data.cpu().numpy()
        else:
            test_y, test_pred_proba, test_yhat = None, None, None
            y_test, y_test_proba, y_test_hat = None, None, None 

        try:
            run_metrics = calculate_metrics(y_val, y_val_hat, y_val_proba=y_val_proba, 
                                            y_test=y_test, y_test_hat=y_test_hat, y_test_proba=y_test_proba)

            mlflow.log_metrics(run_metrics)
        except Exception as e:
            warnings.warn(f"{e}")
            log.warning(f"something went wrong while computing metrics: {e}")
            run_metrics = None

        _end_time = time.time()
        run_time = (_end_time - _start_time)
        
        experiment_tags.update(dict(run_time=run_time))
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)

        # Pytorch log model not working
        # *****************************
        #  mlflow.pytorch.log_model(wrapped_model, model_path, registered_model_name=problem_type)     # <------ use mlflow.pytorch.log_model to log trained sklearn model
        #  print("Model saved in run {}, and registered on {} as a new version of model name {}"
        #       .format(active_run, os.environ['MLFLOW_URI'], problem_type))
        _tmp = {f"artifact/{art_name}": art_val 
                for art_name, art_val in six.iteritems(artifacts)}
        _tmp['model_checkpoint'] =  ckpt_path
        helper.log_artifacts(_tmp, run_id, mlflow_uri=mlflow_setup['mlflow_uri'], delete=True) 

    return (run_id, 
            run_metrics, 
            val_y, val_yhat, val_pred_proba, 
            test_y, test_yhat, test_pred_proba)


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
