#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from lightsaber.trainers.temperature_scaling import _ECELoss
from lightsaber.trainers.helper import setup_mlflow

from functools import partial
import copy
from tempfile import NamedTemporaryFile

import logging
log = logging.getLogger()


class BaseModel(nn.Module, ABC):
    """Docstring for BaseModel. """

    def __init__(self):
        """TODO: to be defined. """
        super(BaseModel, self).__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this CmsTrainer
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-lr', '--learning_rate', default=0.02, type=float)
        parser.add_argument('-bs', '--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser

    def _get_packed_last_time(self, output, lengths=None):
        """
        Return last valid output from packed sequence

        ref: https://blog.nelsonliu.me/2018/01/25/extracting-last-timestep-outputs-from-pytorch-rnns/
        """
        if isinstance(output, T.nn.utils.rnn.PackedSequence):
            # Unpack, with batch_first=True.
            output, lengths = T.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        
        if lengths is None:
            if self.batch_first:
                last_output = output[:, -1, :]
            else:
                last_output = output[-1, :, :]
        else:
            # Extract the outputs for the last timestep of each example
            idx = (T.LongTensor(lengths) - 1).view(-1, 1).expand(
                len(lengths), output.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if output.is_cuda:
                idx = idx.cuda(output.data.get_device())
            # Shape: (batch_size, rnn_hidden_dim)
            last_output = output.gather(
                time_dimension, T.autograd.Variable(idx)).squeeze(time_dimension)
        return last_output


class PyModel(pl.LightningModule):
    def __init__(self, hparams, model,
                 train_dataset, val_dataset,
                 cal_dataset=None, test_dataset=None,
                 collate_fn=None, optimizer=None,
                 loss_func=None, out_transform=None, num_workers=4, **kwargs):
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

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
                           
    def predict_proba(self, *args, **kwargs):
        logit, _ = self.forward(*args, **kwargs)
        pred = self.out_transform(self.temperature_scale(logit))
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
        loss = self.loss_func(y_logits, y)
        
        outputs = self.out_transform(y_logits)
        _, predicted = T.max(outputs.data, 1)
        
        n_correct = (predicted == y).sum().item()
        n_examples = y.size(0)
        return loss, n_correct, n_examples

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_examples = self._common_step(batch, batch_idx)
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

    @pl.data_loader
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

    @pl.data_loader
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

    @pl.data_loader
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

    @pl.data_loader
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

    # This function probably should live outside of this class, but whatever
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
    Function to run supervised training on  cms data

    Parameters
    ----------
    TBD

    Returns
    -------
    trained `base_model` wrapped as `CmsModel`

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
        _start_time = time.time()
        trainer.fit(wrapped_model)

        trainer.restore_weights(wrapped_model) 
        
        mlflow.log_metric('train_score', trainer.callback_metrics['train_score'])
        mlflow.log_metric('train_loss', float(trainer.callback_metrics['loss'].data.cpu().numpy()))
        mlflow.log_metric('val_loss', float(trainer.callback_metrics['val_loss'].data.cpu().numpy()))
        mlflow.log_params(wrapped_model.get_params())
        
        try:
            ckpt_path = trainer.callbacks[0].best_model_path
        except Exception as e:
            ckpt_path = _find_checkpoint(wrapped_model)

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
            y_test, y_test_proba, y_test_hat = None, None, None 

        run_metrics = calculate_metrics(y_val, y_val_hat, y_val_proba=y_val_proba, 
            y_test=y_test, y_test_hat=y_test_hat, y_test_proba=y_test_proba)

        mlflow.log_metrics(run_metrics)

        _end_time = time.time()
        run_time = (_end_time - _start_time)
        
        experiment_tags.update(dict(run_time=run_time))
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)

        # Pytorch log model not working
        # *****************************
        mlflow.log_artifact(ckpt_path, 'model_checkpoint')
        #  mlflow.pytorch.log_model(wrapped_model, model_path, registered_model_name=problem_type)     # <------ use mlflow.pytorch.log_model to log trained sklearn model
        #  print("Model saved in run {}, and registered on {} as a new version of model name {}"
        #       .format(active_run, os.environ['MLFLOW_URI'], problem_type))

        # Other artifacts
        for art_name, art_val in six.iteritems(artifacts):
            with NamedTemporaryFile(delete=False) as tmp_file:
                pickle.dump(art_val, open(tmp_file.name, 'wb'))
                tmp_filename = tmp_file.name
            mlflow.log_artifact(tmp_filename, f"artifact/{art_name}")
            os.remove(tmp_filename)
            log.info(f"Logged {art_name}") 
    return run_metrics, test_y, test_yhat, test_pred_proba
