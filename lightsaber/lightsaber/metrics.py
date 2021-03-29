#!/usr/bin/env python
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
def t2e_mae(risk, T, E):
    '''
    Computes the mean absolute error for the predicted time to event  
    Input: 
        risk: the rist (partial hazard) which is exp(B' X)
        T: time to event
        E: indicator for the event whetheer the event is observed (1) or censored (0)
    Output:
        mae: mean absolute error for the observed events 
    '''
    mae_list = []
    for i in range(T.shape[0]):
        if E[i] == 1:
            mae_list.append(abs(T[i]-risk[i]))
    mae = np.mean(mae_list)
    return mae 

def concordance_score(risk, T, E, include_ties=True, approximate=True):
    '''
    There are two verision of the C-index. One is extremely fast but a very 
    good approximation of the true metric. The other is the true implementation 
    but really really slow for large data. 
    This functions dispatches the call to the required implementation based on 
    the `approximate` variable.
    
    Input:
        risk: the rist (partial hazard) which is exp(B' X)
        T: time to event
        E: indicator for the event whetheer the event is observed (1) or censored (0)
        include_ties: whether to include ties in the computation 
        approximate: whether to call the the fast approximate version or the 
                time-consuming accurate version 
    '''
    if approximate:
        return concordance_index(T, risk, E)
    else:
        return acc_concordance_score(risk, T, E, include_ties)[0]


class _BTree(object):

    """A simple balanced binary order statistic tree to help compute the concordance.
    When computing the concordance, we know all the values the tree will ever contain. That
    condition simplifies this tree a lot. It means that instead of crazy AVL/red-black shenanigans
    we can simply do the following:
    - Store the final tree in flattened form in an array (so node i's children are 2i+1, 2i+2)
    - Additionally, store the current size of each subtree in another array with the same indices
    - To insert a value, just find its index, increment the size of the subtree at that index and
      propagate
    - To get the rank of an element, you add up a bunch of subtree counts
    """

    def __init__(self, values):
        """
        Parameters:
            values: List of sorted (ascending), unique values that will be inserted.
        """
        self._tree = self._treeify(values)
        self._counts = np.zeros_like(self._tree, dtype=int)

    @staticmethod
    def _treeify(values):
        """Convert the np.ndarray `values` into a complete balanced tree.
        Assumes `values` is sorted ascending. Returns a list `t` of the same length in which t[i] >
        t[2i+1] and t[i] < t[2i+2] for all i."""
        if len(values) == 1:  # this case causes problems later
            return values
        tree = np.empty_like(values)
        # Tree indices work as follows:
        # 0 is the root
        # 2n+1 is the left child of n
        # 2n+2 is the right child of n
        # So we now rearrange `values` into that format...

        # The first step is to remove the bottom row of leaves, which might not be exactly full
        last_full_row = int(np.log2(len(values) + 1) - 1)
        len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = np.s_[:2 * len_ragged_row:2]
            tree[-len_ragged_row:] = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)

        # Now `values` is length 2**n - 1, so can be packed efficiently into a tree
        # Last row of nodes is indices 0, 2, ..., 2**n - 2
        # Second-last row is indices 1, 5, ..., 2**n - 3
        # nth-last row is indices (2**n - 1)::(2**(n+1))
        values_start = 0
        values_space = 2
        values_len = 2 ** last_full_row
        while values_start < len(values):
            tree[values_len - 1:2 * values_len - 1] = values[values_start::values_space]
            values_start += int(values_space / 2)
            values_space *= 2
            values_len = int(values_len / 2)
        return tree

    def insert(self, value):
        """Insert an occurrence of `value` into the btree."""
        i = 0
        n = len(self._tree)
        while i < n:
            cur = self._tree[i]
            self._counts[i] += 1
            if value < cur:
                i = 2 * i + 1
            elif value > cur:
                i = 2 * i + 2
            else:
                return
        raise ValueError("Value %s not contained in tree."
                         "Also, the counts are now messed up." % value)

    def __len__(self):
        return self._counts[0]

    def rank(self, value):
        """Returns the rank and count of the value in the btree."""
        i = 0
        n = len(self._tree)
        rank = 0
        count = 0
        while i < n:
            cur = self._tree[i]
            if value < cur:
                i = 2 * i + 1
                continue
            elif value > cur:
                rank += self._counts[i]
                # subtract off the right tree if exists
                nexti = 2 * i + 2
                if nexti < n:
                    rank -= self._counts[nexti]
                    i = nexti
                    continue
                else:
                    return (rank, count)
            else:  # value == cur
                count = self._counts[i]
                lefti = 2 * i + 1
                if lefti < n:
                    nleft = self._counts[lefti]
                    count -= nleft
                    rank += nleft
                    righti = lefti + 1
                    if righti < n:
                        count -= self._counts[righti]
                return (rank, count)
        return (rank, count)

def concordance_index(event_times, predicted_event_times, event_observed=None):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.
    The concordance index is a value between 0 and 1 where,
    0.5 is the expected result from random predictions,
    1.0 is perfect concordance and,
    0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)
    Score is usually 0.6-0.7 for survival models.
    See:
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.
    Parameters:
      event_times: a (n,) array of observed survival times.
      predicted_event_times: a (n,) array of predicted survival times.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.
    Returns:
      c-index: a value between 0 and 1.
    """
    event_times = np.array(event_times, dtype=float)
    predicted_event_times = np.array(predicted_event_times, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1 or
                                  event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if (predicted_event_times.ndim == 2 and
        (predicted_event_times.shape[0] == 1 or
         predicted_event_times.shape[1] == 1)):
        # Flatten array
        predicted_event_times = predicted_event_times.ravel()

    if event_times.shape != predicted_event_times.shape:
        raise ValueError("Event times and predictions must have the same shape")
    if event_times.ndim != 1:
        raise ValueError("Event times can only be 1-dimensional: (n,)")

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")
        event_observed = np.array(event_observed, dtype=float).ravel()

    def _concordance_index_internal(event_times, predicted_event_times, event_observed):
        """Find the concordance index in n * log(n) time.
        Assumes the data has been verified by lifelines.utils.concordance_index first.
        """
        # Here's how this works.
        #
        # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
        # would be to iterate over the cases in order of their true event time (from least to greatest),
        # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
        # cases that we know should be ranked lower than the case we're looking at currently).
        #
        # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
        # value less than x"), then the following algorithm is n log n:
        #
        # Sort the times and predictions by time, increasing
        # n_pairs, n_correct := 0
        # pool := {}
        # for each prediction p:
        #     n_pairs += len(pool)
        #     n_correct += rank(pool, p)
        #     add p to pool
        #
        # There are three complications: tied ground truth values, tied predictions, and censored
        # observations.
        #
        # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
        # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
        # simultaneously at the end.
        #
        # - To handle tied predictions, which should each count for 0.5, we switch to
        #     n_correct += min_rank(pool, p)
        #     n_tied += count(pool, p)
        #
        # - To handle censored observations, we handle each batch of tied, censored observations just
        # after the batch of observations that died at the same time (since those censored observations
        # are comparable all the observations that died at the same time or previously). However, we do
        # NOT add them to the pool at the end, because they are NOT comparable with any observations
        # that leave the study afterward--whether or not those observations get censored.

        died_mask = event_observed.astype(bool)
        # TODO: is event_times already sorted? That would be nice...
        died_truth = event_times[died_mask]
        ix = np.argsort(died_truth)
        died_truth = died_truth[ix]
        died_pred = predicted_event_times[died_mask][ix]

        censored_truth = event_times[~died_mask]
        ix = np.argsort(censored_truth)
        censored_truth = censored_truth[ix]
        censored_pred = predicted_event_times[~died_mask][ix]

        censored_ix = 0
        died_ix = 0
        times_to_compare = _BTree(np.unique(died_pred))
        num_pairs = 0
        num_correct = 0
        num_tied = 0

        def handle_pairs(truth, pred, first_ix):
            """
            Handle all pairs that exited at the same time as truth[first_ix].
            Returns:
            (pairs, correct, tied, next_ix)
            new_pairs: The number of new comparisons performed
            new_correct: The number of comparisons correctly predicted
            next_ix: The next index that needs to be handled
            """
            next_ix = first_ix
            while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
                next_ix += 1
            pairs = len(times_to_compare) * (next_ix - first_ix)
            correct = 0
            tied = 0
            for i in range(first_ix, next_ix):
                rank, count = times_to_compare.rank(pred[i])
                correct += rank
                tied += count

            return (pairs, correct, tied, next_ix)

        # we iterate through cases sorted by exit time:
        # - First, all cases that died at time t0. We add these to the sortedlist of died times.
        # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
        #   comparable to subsequent elements.
        while True:
            has_more_censored = censored_ix < len(censored_truth)
            has_more_died = died_ix < len(died_truth)
            # Should we look at some censored indices next, or died indices?
            if has_more_censored and (not has_more_died
                                    or died_truth[died_ix] > censored_truth[censored_ix]):
                pairs, correct, tied, next_ix = handle_pairs(censored_truth, censored_pred, censored_ix)
                censored_ix = next_ix
            elif has_more_died and (not has_more_censored
                                    or died_truth[died_ix] <= censored_truth[censored_ix]):
                pairs, correct, tied, next_ix = handle_pairs(died_truth, died_pred, died_ix)
                for pred in died_pred[died_ix:next_ix]:
                    times_to_compare.insert(pred)
                died_ix = next_ix
            else:
                assert not (has_more_died or has_more_censored)
                break

            num_pairs += pairs
            num_correct += correct
            num_tied += tied

        if num_pairs == 0:
            raise ZeroDivisionError("No admissable pairs in the dataset.")

        return (num_correct + num_tied / 2) / num_pairs
        
    return _concordance_index_internal(event_times,
                              predicted_event_times,
                              event_observed)

def acc_concordance_score(risk, T, E, include_ties=True):
    '''
    risk [n_sample] : the predicted risk
    T [n_sample]    : the ground truth time 2 event 
    E [n_sample]    : the ground truth event indicator 
    results         : dictionary 
                        results[0] = C-index;
                        results[1] = nb_pairs;
                        results[2] = nb_concordant_pairs;
    '''
    # Ordering risk, T and E in descending order according to T
    order = np.argsort(-T)
    risk = risk[order]
    T = T[order]
    E = E[order]

    # Calculating the c-index
    results = _concordance_index(risk, T, E, include_ties) # this is a C++ implementation which is fast 

    return results

def accuracy_pu(y_true, y_hat):
    '''
    Accuracy for PU 
    Input: 
        y_true: (1d np.array) actual labels, C.LABEL_POSITIVE - positve class, C.LABEL_NEGATIVE - negative class
        y_hat: (1d np.array) predicted labels, C.LABEL_POSITIVE - positve class, C.LABEL_NEGATIVE - negative class
    Output:
        accuracy: accuracy score for PU data  
    '''
    prior = sum(y_true==C.LABEL_POSITIVE) / y_true.shape[0] # prior 
    yhat_p, yhat_u = y_hat[y_true == C.LABEL_POSITIVE], y_hat[y_true == C.LABEL_NEGATIVE] # prediction for each class 
    n_p, n_u = np.sum(y_true == C.LABEL_POSITIVE), np.sum(y_true == C.LABEL_NEGATIVE) # number of positive and unlabeled examples 
    f_n, f_p = np.sum(yhat_p == C.LABEL_NEGATIVE) / n_p, np.sum(yhat_u == C.LABEL_POSITIVE) / n_u # false negative/positive rates
    pu_risk = prior * f_n + np.maximum(0, f_p + prior * f_n - prior) # error / empirical risk 
    accuracy = 1 - pu_risk # accuracy 
    return accuracy 

def f1_pu(y_true, y_hat):
    '''
    F1 score criterion that has the property that it is high when precision and recall are both high.
    This is not the traditional F1 score `(2*p*r) / (p+r)` but it is adapted for PU data.
    It is based on 
    "
    Bekker, J., & Davis, J. (2020). 
    Learning from positive and unlabeled data: a survey. 
    Machine Learning, 109, 719-760.
    "
    Input: 
        y_true: (1d np.array) actual labels, C.LABEL_POSITIVE - positve class, C.LABEL_NEGATIVE - negative class
        y_hat: (1d np.array) predicted labels, C.LABEL_POSITIVE - positve class, C.LABEL_NEGATIVE - negative class
    Output:
        f1_pu: f1 score for PU data  
    '''
    recall = recall_score(y_true=y_true,y_pred=y_hat)
    prevalence_hat = sum(y_hat==1) / y_hat.size
    f1_pu = recall**2 / prevalence_hat
    return f1_pu 


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
    merged_values = np.array([y_true, y_hat, y_proba,]).T #np.hstack((y_true, probas_pred)) # concat y_true and prediction probabilities
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
    def __init__(self, mode='classifier'):
        super(Metrics, self).__init__()
        self.mode = mode

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
