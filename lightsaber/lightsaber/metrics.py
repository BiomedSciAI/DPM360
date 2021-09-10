#!/usr/bin/env python
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
import six
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, 
                             average_precision_score, precision_score, 
                             recall_score, brier_score_loss)
import numpy as np
from lightsaber import constants as C

import logging
log = logging.getLogger()

try:
    from pysurvival.utils._metrics import _concordance_index
except ImportError:
    log.warning("pysurvival not installed... survival models wont work")

'''
The concordance_score metric is added. Accuract but time consuming

concordance_index is fast approximate c-index. It could be used for 
optimizing the model but may be not the final reported one 

TODO

- Brier score

'''


# **************************************************************************
#  library of metric functions 
# **************************************************************************


def pr_at_k(y_true, y_hat, y_proba, pct, average='binary'):
    '''
     Calculate precision and recall @ k
    Args:
        y_true: (1d np.array) actual labels, 1 - positve class, 0 - negative class
        y_hat: (1d np.array) predicted labels, 1 - positve class, 0 - negative class
        y_proba: (1d np.array) probability in positive class
        k: (int) number of top (highest probability) predictions
        average: averaging method (see doc for sklearn.metrics.precision_score)
    Returns:
        dict with precision_at_k and recall_at_k
    '''
    k = round(pct * len(y_true))
    merged_values = np.vstack([y_true.T, y_hat.T, y_proba.T]).T #np.hstack((y_true, probas_pred)) # concat y_true and prediction probabilities
    merged_values = merged_values[(-merged_values[:,2]).argsort()] # sort by probabilities
    top_k_true_proba = merged_values[:k,:] # select top k with highest probabilities
    y_true_top_k = top_k_true_proba[:,0] # seperate y_true, y_hat, preds for clarity
    y_hat_top_k = top_k_true_proba[:,1]

    precision = precision_score(y_true = y_true_top_k, y_pred=y_hat_top_k, average=average)
    recall = recall_score(y_true = y_true_top_k, y_pred=y_hat_top_k, average=average)
    return {('precision_at_' + str(pct * 100) + 'pct_' + str(k)): precision,
            ('recall_at_'+ str(pct * 100) + 'pct_' +str(k)): recall}


# **************************************************************************
#  Main interface for other modules
# **************************************************************************
class Metrics(object):
    """docstring for Metrics"""
    __supported_modes = ['classifier'] #, 'pu_classifier', 't2e']

    def __init__(self, mode='classifier'):
        super(Metrics, self).__init__()
        self.mode = mode
        if self.mode not in self.__supported_modes:
            raise NotImplementedError(f'modes outside {self.__supported_modes} not yet supported')

    def __call__(self, *args, **kwargs):
        if self.mode == 'classifier':
            ret_val = self.classifier_metrics(*args, **kwargs)
        elif self.mode == 'pu_classifier':
            ret_val = self.pu_classifier_metrics(*args, **kwargs)
        elif self.mode == 't2e':
            ret_val = self.t2e_metrics(*args, **kwargs)
        return ret_val

    def __repr__(self):
        s = f"Metrics[{self.mode}]"
        return s
    
    @staticmethod
    def classifier_metrics(y_val, y_val_hat, y_val_proba=None, val_score=None,
                           y_test=None, y_test_hat=None, y_test_proba=None, test_score=None):
        '''
        Calculate metrics for model evaluation 
        Args:
            y_true: (1d np.array) actual labels, 1 - positve class, 0 - negative class
            y_hat: (1d np.array) predicted labels, 1 - positve class, 0 - negative class
            y_proba: (1d np.array) probability in positive class
        Returns:
            dict with train_error, test_error, precision, recall, auc, auprc, accuracy, and precision recalls at k%
        '''
        # Validation part  
        val_precision = precision_score(y_true=y_val, y_pred=y_val_hat)
        val_recall = recall_score(y_true=y_val, y_pred=y_val_hat)
        val_accuracy = accuracy_score(y_true=y_val, y_pred=y_val_hat)
        val_auc, val_auprc, val_brier_score = 0, 0, 0
        _prak = {}
        if y_val_proba is not None:
            val_auc = roc_auc_score(y_val, y_val_proba)
            val_auprc = average_precision_score(y_val, y_val_proba)
            val_brier_score = brier_score_loss(y_val, y_val_proba)
            # Recall on highest ½%, 1%, 2%, 5 of risk scores
            for pct in [0.005, 0.01, 0.02, 0.05]: 
                _tmp = pr_at_k(y_true=y_val, y_hat=y_val_hat,
                               y_proba=y_val_proba, pct=pct)
                for key, value in six.iteritems(_tmp):
                    _prak[f'Val_{key}'] = value 

            metrics = {
                'Val_Precision': val_precision,
                'Val_Recall': val_recall,
                'Val_AUCROC': val_auc,
                'Val_AUPRC': val_auprc,
                'Val_Accuracy': val_accuracy,
                'Val_Brier_score': val_brier_score,
            }
            metrics.update(_prak) #**pr_at_5_pct, **pr_at_10pct, **pr_at_25pct, **pr_at_50pct}
        if val_score is not None:
            val_error = 1 - val_score
            metrics['Val_error'] = val_error
        else:
            val_error = None
        
        # Test part  
        if y_test is not None:
            test_precision = precision_score(y_true=y_test, y_pred=y_test_hat)
            test_recall = recall_score(y_true=y_test, y_pred=y_test_hat)
            test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_hat)
            test_auc, test_auprc, test_brier_score = 0, 0, 0
            _prak = {}
            if y_test_proba is not None:
                test_auc = roc_auc_score(y_test, y_test_proba)
                test_auprc = average_precision_score(y_test, y_test_proba)
                test_brier_score = brier_score_loss(y_test, y_test_proba)
                # Recall on highest ½%, 1%, 2%, 5 of risk scores
                for pct in [0.005, 0.01, 0.02, 0.05]: 
                    _tmp = pr_at_k(y_true=y_test, y_hat=y_test_hat,
                                   y_proba=y_test_proba, pct=pct)
                    for key, value in six.iteritems(_tmp):
                        _prak[f'Test_{key}'] = value 

                metrics.update({
                    'Test_Precision': test_precision,
                    'Test_Recall': test_recall,
                    'Test_AUCROC': test_auc,
                    'Test_AUPRC': test_auprc,
                    'Test_Accuracy': test_accuracy,
                    'Test_Brier_score': test_brier_score,
                })
                metrics.update(_prak) #**pr_at_5_pct, **pr_at_10pct, **pr_at_25pct, **pr_at_50pct}

            if test_score is not None:
                test_error = 1 - test_score
                metrics['Test_error'] = test_error
            else:
                test_error = None
        return metrics

    @staticmethod
    def pu_classifier_metrics(y_val, y_val_hat, y_val_proba=None, val_score=None,
                           y_test=None, y_test_hat=None, y_test_proba=None, test_score=None):
        '''
        Calculate metrics for model evaluation 
        Args:
            y_true: (1d np.array) actual labels, 1 - positve class, 0 - negative class
            y_hat: (1d np.array) predicted labels, 1 - positve class, 0 - negative class
            y_proba: (1d np.array) probability in positive class
        Returns:
            dict with train_error, test_error, precision, recall, auc, auprc, accuracy, and precision recalls at k%
        '''

        
        val_precision = precision_score(y_true=y_val, y_pred=y_val_hat)
        val_recall = recall_score(y_true=y_val,y_pred=y_val_hat)
        val_accuracy = accuracy_score(y_true=y_val, y_pred=y_val_hat)
        val_f1_score_pu = f1_pu(y_val, y_val_hat)
        val_accuracy_score_pu = accuracy_pu(y_val, y_val_hat)

        metrics = {
            'Val_Precision': val_precision,
            'Val_Recall': val_recall,
            'Val_Accuracy': val_accuracy,
            'Val_F1_PU': val_f1_score_pu,
            'Val_Accuracy_PU': val_accuracy_score_pu
        }
        if val_score is not None:
            val_error = 1 - val_score
            metrics['Val_error'] = val_error
        else:
            val_error = None

        if y_test is not None:
            test_precision = precision_score(y_true=y_test, y_pred=y_test_hat)
            test_recall = recall_score(y_true=y_test,y_pred=y_test_hat)
            test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_hat)
            test_f1_score_pu = f1_pu(y_test, y_test_hat)
            test_accuracy_score_pu = accuracy_pu(y_test, y_test_hat)

            metrics.update({
                            'Test_Precision': test_precision,
                            'Test_Recall': test_recall,
                            'Test_Accuracy': test_accuracy,
                            'Test_F1_PU': test_f1_score_pu,
                            'Test_Accuracy_PU': test_accuracy_score_pu
                           })
            if test_score is not None:
                test_error = 1 - test_score
                metrics['Test_error'] = test_error
            else:
                test_error = None
        return metrics

    @staticmethod
    def t2e_metrics(*args, **kwargs):
        raise NotImplementedError()
        pass
