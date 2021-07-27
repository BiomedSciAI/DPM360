#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six
import pickle

import mlflow
import mlflow.sklearn
import torch as T
import tqdm
import time
from copy import deepcopy

from lightsaber import constants as C
from lightsaber.trainers import pt_trainer
from lightsaber.trainers import sk_trainer
from lightsaber.metrics import concordance_score, t2e_mae
from lightsaber.trainers.helper import (setup_mlflow, get_model, get_predefined_split,
                                        load_sk_model, save_sk_model, import_model_class)


import logging
log = logging.getLogger()

def calculate_metrics(prediction, target, event, approximate=False, fold="test"):
    _c_index = concordance_score(prediction.reshape(-1,), 
                                 target.reshape(-1,), 
                                 event.reshape(-1,), 
                                 include_ties=True, 
                                 approximate=approximate)
    _t_mae = t2e_mae(prediction,
                       target,
                       event)
    metrics = {f"{fold}_c_index": _c_index,
               f"{fold}_t_mae": _t_mae}
    
    return metrics

class SurvModel(pt_trainer.PyModel):
    def _common_step(self, batch, batch_idx):
        # import ipdb; ipdb.set_trace();
        if len(batch) == 5:
            x, y, e, lengths, idx = batch
            y_risk = self.forward(x, lengths=lengths)
        else:
            raise NotImplementedError()
        if (len(y_risk.shape) == 3) and (len(y.shape) != 3):
            y_risk.squeeze_(1) # squeeze time dimension
        loss = self.loss_func(y_risk, y, e)
        
        # outputs = self.out_transform(y_logits)
        # _, predicted = T.max(outputs.data, 1)
        
        n_correct = 0 #(predicted == y).sum().item()
        n_examples = y.size(0)
        return loss, n_correct, n_examples    
    
    def training_step(self, batch, batch_idx):
        # import ipdb; ipdb.set_trace();
        loss, n_correct, n_examples = self._common_step(batch, batch_idx)
        if hasattr(self.model, 'apply_regularization'):
            loss += self.model.apply_regularization()
            
        # train_score = n_correct / n_examples
        tensorboard_log = {'batch_train_loss': loss} #, 'batch_train_score': train_score}
        return dict(loss=loss, 
                    # train_score=train_score,
                    log=tensorboard_log) #, train_n_correct=n_correct, 

    def predict(self, *args, **kwargs):
        """Monkey patching with Classification model
        """
        if hasattr(self.model, 'predict'):
            pred = self.model.predict(*args, **kwargs)
        else:
            pred = super().predict_proba(*args, **kwargs)
        if len(pred.shape) == 3:
            pred.squeeze_(2) # squeeze the last dimension
        return pred

    
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
        all_e = []
        
        for (bIdx, batch) in tqdm.tqdm(enumerate(dataloader), total=n_batches):
            if bIdx == n_batches:
                break
            
            # making it compatible with non trainer run
            try:       
                if self.trainer.on_gpu:
                    batch = self.trainer.transfer_batch_to_gpu(batch, self.trainer.root_gpu)
            except Exception:
                pass
                
            if len(batch) == 5:
                x, y, e, lengths, idx = batch
                # y_risk = self.forward(x, lengths=lengths)           
                # x.to(device)
                # summary.to(device)
                # y.to(device)
                
                pred = self.predict(x, lengths) #, summary)
                
            # avoid memory leak when storing too much data from a dataloader
            # ref: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
            all_preds.append(deepcopy(pred.detach()))
            all_y.append(deepcopy(y.detach()))
            all_e.append(deepcopy(e.detach()))
        
        # import ipdb; ipdb.set_trace();
        preds = T.cat(all_preds, dim=0)
        y = T.cat(all_y, dim=0)
        e = T.cat(all_e, dim=0)
        
        #_, yhat = T.max(preds, dim=1)
        
        # _n_correct = (yhat == y).sum().item()
        # _n_examples = y.size(0)
        # score = (_n_correct / _n_examples)

        self.to(_orig_device)
        return preds, y, e #, score
    

# *************************************************************************************
#                     MLFlow Helpers for Survival Models
#
# *************************************************************************************
def log_sk_model(model, metrics, experiment_name, 
                 experiment_tags=None,
                 params=None,
                 mlflow_uri=C.MLFLOW_URI):
    
    mlflow_setup = setup_mlflow(experiment_name=experiment_name,
                                mlflow_uri=mlflow_uri)
    
    with mlflow.start_run():
        try:
            mlflow.sklearn.log_model(model, 'model')
        except TypeError as e:
            log.warning(f"Error while logging model, artifacting instead. {e}")
            _fname = tempfile.NamedTemporaryFile(delete=False)
            _fname.close()
            save_sk_model(model, _fname.name)
            mlflow.log_artifact(_fname.name, 'model')
            os.unlink(_fname.name)

        # use sklearn native API for get params
        if params is None:
            params = model.get_params()    
        
        # Other logs
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)
            
        log.debug(f"Model logged to mlflow experiment: {experiment_name}")
    return

def run_sk_training_with_mlflow(mlflow_conf, 
                                sk_model,
                                train_dataloader, 
                                val_dataloader=None, 
                                test_dataloader=None,
                                **kwargs):
    tune = kwargs.pop('tune', False)

    model_path = kwargs.get('model_path', 'model')
    artifacts = kwargs.pop('artifacts', dict())
    
    mlflow_conf.setdefault('problem_type', 't2e')
    mlflow_setup = setup_mlflow(**mlflow_conf)
    
    experiment_name = mlflow_setup['experiment_name']
    
    experiment_setup = kwargs.get('experiment_setup', None)
    experiment_setup.setdefault('approximate', False)

    

    model_save_dir = Path(kwargs.get('model_save_dir', C.MODEL_SAVE_DIR))
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    (X_train, T_train, E_train) = (train_dataset.data.values, 
                                   train_dataset.target.values.reshape(-1,), 
                                   train_dataset.event.values.reshape(-1,))
    
    if val_dataloader is not None:
        raise NotImplementedError()
        (X_val, T_val, E_val) = (val_dataset.data.values, 
                                 val_dataset.target.values.reshape(-1,), 
                                 val_dataset.event.values.reshape(-1,))
        X_val, y_val = val_dataloader.read_data()
        # TODO: extend predefined split to three 
        cv, _X, _T, _E = get_predefined_split(X_train, T_train, E_train, 
                                          X_val, T_val, E_val)
    else:
        cv = kwargs.get('cv', C.DEFAULT_CV)
        _X = X_train
        _T = T_train
        _E = E_train

    if test_dataloader is not None:
        (X_test, T_test, E_test) = (test_dataset.data.values, 
                                    test_dataset.target.values.reshape(-1,), 
                                    test_dataset.event.values.reshape(-1,))

    print(mlflow_setup)

    with mlflow.start_run():
        # mlflow.sklearn.log_model(sk_model.model, experiment_name)
        # mlflow.log_params(sk_model.model.get_params())
        _start_time = time.time()

        if tune:
            raise NotImplementedError()
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
            sk_model.fit(X_train, T_train, E_Train)#, Xstd = X_train_std)
        
            mlflow.sklearn.log_model(sk_model.model, experiment_name + '_model')
            mlflow.log_params(sk_model.params)
        
            print(f"*** Experiment: {experiment_name}({experiment_setup}) has finished training. ***")
            save_sk_model(sk_model, model_save_dir + '/' + 'experiment_name')
        

        # for split_id, (train_index, val_index) in enumerate(cv):
        #     _X_train, _X_val = _X[train_index,:], X[test_index,:]
        #     _y_train, _y_val = _y[train_index], _y[test_index]
        #     
        #     y_val_proba = sk_model.predict_proba(X_val)
        #     if y_val_proba.ndim > 1:
        #         y_val_proba = y_val_proba[:,1]
        #  
        #     y_val_hat = sk_model.predict(X_val)
        #     val_score = sk_model.score(X_val, y_val)
        
        metrics = dict()
        payload = dict(model=sk_model)
        
        if val_dataloader is not None:
            pred_val = sk_model.predict(X_val)
            payload.update(dict(T_val=T_val,
                                E_val=E_val,
                                pred_val=pred_val)) 
            metrics.update(calculate_metrics(pred_val, T_val, E_val, fold="val", 
                                             approximate=experiment_setup['approximate']))

        if test_dataloader is not None:
            pred_test = sk_model.predict(X_test)
            
            payload.update(dict(T_test=T_test,
                                E_test=E_test,
                                pred_test=pred_test))
            metrics.update(calculate_metrics(pred_test, T_test, E_test, fold="test", 
                                             approximate=experiment_setup['approximate']))

        # Calculate metrics
        sk_model.metrics = metrics
        # log metrics
        mlflow.log_metrics(sk_model.metrics)
        print(sk_model.metrics)
        
        _end_time = time.time()
        run_time = (_end_time - _start_time)
        experiment_tags = dict(#approximate=approximate, #already present in experiment setup
                               run_time=run_time)
        experiment_tags.update(**experiment_setup)
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)

        # Other artifacts
        for art_name, art_val in six.iteritems(artifacts):
            with NamedTemporaryFile(delete=False) as tmp_file:
                pickle.dump(art_val, open(tmp_file.name, 'wb'))
                tmp_filename = tmp_file.name
            mlflow.log_artifact(tmp_filename, f"artifact/{art_name}")
            os.remove(tmp_filename)
            log.info(f"Logged {art_name}") 
        return payload

def run_pt_training_with_mlflow(mlflow_conf, trainer, wrapped_model, **kwargs):
    """
    Function to run supervised training on  cms data

    Parameters
    ----------
    TBD

    Returns
    -------
    trained `base_model` wrapped as `CmsModel`

    """
    tune = kwargs.pop('tune', False)

    model_path = kwargs.get('model_path', 'model')
    artifacts = kwargs.pop('artifacts', dict())
    
    mlflow_conf.setdefault('problem_type', 't2e')
    mlflow_setup = setup_mlflow(**mlflow_conf)
    
    experiment_name = mlflow_setup['experiment_name']
    
    experiment_tags = dict(approximate=False)
    experiment_tags.update(**kwargs)

    print(mlflow_setup)

    with mlflow.start_run():
        _start_time = time.time()
        trainer.fit(wrapped_model)
        
        #  import ipdb; ipdb.set_trace();
        trainer.restore_weights(wrapped_model) 
        wrapped_model.eval()

        mlflow.log_metric('train_loss', float(trainer.callback_metrics['loss'].data.cpu().numpy()))
        mlflow.log_metric('val_loss', float(trainer.callback_metrics['val_loss'].data.cpu().numpy()))
        mlflow.log_params(wrapped_model.get_params())

        test_dataloader = wrapped_model.test_dataloader()
        # test_dataloader = DataLoader(wrapped_model.test_dataset, collate_fn=wrapped_model.collate_fn,
        #                              batch_size=wrapped_model.bk_hparams.batch_size)

        (test_pred, 
         test_y, 
         test_e) = wrapped_model.run(test_dataloader, 
                                     overfit_pct=kwargs.get('overfit_pct', 0))
        
        T_test=test_y.data.cpu().numpy()
        E_test=test_e.data.cpu().numpy()
        pred_test=test_pred.data.cpu().numpy()
        
        test_metrics = calculate_metrics(pred_test, T_test, E_test, 
                                         fold="test", 
                                         approximate=experiment_tags['approximate'])

        mlflow.log_metrics(test_metrics)

        _end_time = time.time()
        run_time = (_end_time - _start_time)
        
        experiment_tags.update(dict(run_time=run_time))
        if experiment_tags is not None:
            mlflow.set_tags(experiment_tags)
        
        # Pytorch log model not working
        # *****************************
        ckpt_path = trainer.callbacks[0].best_model_path
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
    return test_metrics, T_test, E_test, pred_test
