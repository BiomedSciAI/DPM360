#!/usr/bin/env python
from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace
from functools import partial
from inspect import signature
from tempfile import NamedTemporaryFile
import copy
import tqdm
import os
import re

from typing import Optional, Callable

import torch as T
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch

from torchmetrics.functional import accuracy

from lightsaber import constants as C
from .regularizer import l1_regularization, l2_regularization

import warnings
import logging
log = logging.getLogger()


def load_model(model, ckpt_path):
    checkpoint = pl.utilities.cloud_io.load(ckpt_path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'on_load_checkpoint'): 
        model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model


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


# *****************************************************
#                 Task Specific
# *****************************************************
class BaseTask(pl.LightningModule, ABC):
    """BaseTask"""
    def __init__(self, 
                 hparams:Namespace, 
                 model:nn.Module,
                 optimizer: Optional[Optimizer] = None,
                 loss_func: Optional[Callable] = None, 
                 out_transform: Optional[Callable] = None, 
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
        optimizer: torch.optim.Optimizer, optional
            pytorch optimizer. If not provided, Adam is used with standard parameters
        loss_func: callable
            if provided, used to compute the loss. Default: cross entropy loss
        out_transform: callable
            if provided, convert logit to expected format. Default, softmax
        kwargs: dict, optional
            other parameters accepted by pl.LightningModule
        """
        super(BaseTask, self).__init__()
        #  self.bk_hparams = hparams
        self.model = model

        self._debug = debug
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._scheduler = kwargs.get('scheduler', None)
        self._kwargs = kwargs

        # save hyper-parameters
        self.save_hyperparameters(hparams)

        self._setup_task()
        return

    @abstractmethod
    def _setup_task(self):
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
    @abstractmethod
    def _process_common_output(self, y_pred):
        y_hat = None
        return y_hat
    
    # TODO: make this an abstractmethod. currently done for classification
    @abstractmethod
    def _calculate_score(self, y_pred, y):
        score = None
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

# *****************************************************
#             Model specific 
# *****************************************************
class BaseModel(nn.Module, ABC):
    """Docstring for BaseModel. """

    def __init__(self):
        """TODO: to be defined. """
        super(BaseModel, self).__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this model
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


class ClassifierMixin(object):
    def get_logit(self):
        bias = self._kwargs.get('op_bias', False)
        return nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

