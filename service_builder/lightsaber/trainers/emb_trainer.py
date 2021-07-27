#! /usr/bin/env python
from lightsaber.trainers.pt_trainer import PyModel
from lightsaber.trainers.t2e_trainer import SurvModel
import tqdm
import torch as T


class EmbModel(PyModel): 
    
    def _common_step(self, batch, batch_idx):
        x, _, lengths, idx = batch 
        x_recons = self.forward(x, lengths=lengths)
        loss = self.loss_func(x_recons, x, lengths=lengths)
        
        n_correct = 0 #(predicted == y).sum().item()
        n_examples = x.size(0)
        return loss, n_correct, n_examples    
    
    def training_step(self, batch, batch_idx):
        """
        TODO: it might be better to move this function up to PyModel 
        as it is generic 
        """
        loss, _, _ = self._common_step(batch, batch_idx)
        if hasattr(self.model, 'apply_regularization'):
            loss += self.model.apply_regularization()
        
        tensorboard_log = {'batch_train_loss': loss} 
        return dict(loss=loss, 
                    log=tensorboard_log) 

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

        n_batches = len(dataloader)
        if overfit_pct > 0:
            n_batches = int(n_batches * overfit_pct)
            
        all_preds = []
        
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
                x, _, lengths, idx = batch 
                pred = self.predict(x, lengths)
                
            # avoid memory leak when storing too much data from a dataloader
            # ref: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
            all_preds.append(deepcopy(pred.detach()))
        
        preds = T.cat(all_preds, dim=0)

        return preds
    
    
