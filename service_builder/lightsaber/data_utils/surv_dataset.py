#!/usr/bin/env python
from __future__ import absolute_import
import pandas as pd
import numpy as np
import torch as T

from lightsaber import constants as C
from lightsaber.data_utils import pt_dataset as ptd
from lightsaber.data_utils import sk_dataloader as skd


class SurvivalDataset(ptd.BaseDataset):
    """Docstring for SurvivalDataset. """

    def __init__(self, tgt_file, feat_file, tgt_name,
                 idx_col, tgt_col, t2e_col,
                 feat_columns=None, time_order_col=None,
                 category_map=C.DEFAULT_MAP,
                 transform=ptd.DEFAULT_TRANSFORM,
                 filter=ptd.DEFAULT_FILTER,
                 device='cpu'
                ):
        """Survival Dataset

        Parameters
        ----------
        tgt_file:
            target file path
        feat_file:
            feature file path
        tgt_name:
            name of the target column
        idx_col:
            index columns
        tgt_col: 
            values in target name that will be used for t2e modeling
        t2e_col:
            time to event column name
        feat_columns:
            feature columns to select from. either list of columns (partials columns using `*` allowed) or a single regex
            Default: `None` -> implies all columns
        time_order_col:
            column(s) that signify the time ordering for a single example.
            Default: `None` -> implies no columns 
        category_map:
            dictionary of column maps
        transform: single callable or list/tuple of callables
            how to transform data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: drop `lightsaber.constants::DEFAULT_DROP_COLS` and fillna
        filter: single callable or list/tuple of callables
            how to filter data. if list of callables provided eg `[f, g]`, `g(f(x))` used 
            Default: no operation
        device: str
            valid pytorch device. `cpu` or `gpu`


        Example
        -------
        idx_col = 'ENROLID'
        tgt_name = 'Stage'
        tgt_col = ['202', '201']
        t2e_col = 'T2E'
        feat_cols = ['AGE_*', 'SEX', 'CCS_*']
        """
        self._tgt_name = tgt_name
        self._t2e_col = t2e_col

        super().__init__(tgt_file=tgt_file, feat_file=feat_file, 
                         idx_col=idx_col, tgt_col=tgt_col, 
                         feat_columns=feat_columns, 
                         time_order_col=time_order_col,
                         category_map=category_map,
                         transform=transform,
                         filter=filter,
                         device=device)

        return

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            data_t, target_t, event_t, _, _ = self.__getitem__(0)
            feat_dim = data_t.shape[-1]
            try:
                target_dim = target_t.shape[-1]
            except Exception:
                target_dim = 1
            try:
                event_dim = event_t.shape[-1]
            except Exception:
                event_dim = 1
            self._shape = (feat_dim, target_dim, event_dim)
        return self._shape

    def read_data(self):
        tgt_file, feat_file = self._tgt_file, self._feat_file
        self.target = pd.read_csv(tgt_file)
        # Filling the target column with the censored name
        self.target[self._tgt_name] = (self.target[self._tgt_name]
                                       .fillna(C.DEFAULT_CENSORED_NAME)
                                       .astype(str))
        
        # pivoting the table to have a single row for each unique value of tgt_col
        self.target = self.target.pivot_table(columns=self._tgt_name, 
                                              values=self._t2e_col,
                                              index=self._idx_col)
        # import ipdb; ipdb.set_trace();

        _event = self.target[C.DEFAULT_CENSORED_NAME]
        self.target = self._select_features(self.target, self._tgt_col)
        _map = dict({True: C.LABEL_CENSORED,
                     False: C.LABEL_OBSERVED})
        self.event = self.target.isnull().applymap(_map.get)
        if isinstance(self.event, pd.Series):
            self.event = self.event.to_frame()
            
        for col in self.target.columns:
            self.target[col].fillna(_event, inplace=True)

        self.data = pd.read_csv(feat_file).set_index(self._idx_col)
        self.data = self.data.loc[self.target.index, :]   # accounting for the option that target can have lesser number of index than data
        self.data = self._select_features(self.data, self._feat_columns)        
        return

    def __getitem__(self, i):
        device = self.device
        data_t, target_t, length, idx = super().__getitem__(i)
        # import pdb; pdb.set_trace()
        # target_t.unsqueeze_(0)
        event = self.event.loc[idx]
        event_t = T.LongTensor(event).to(device)
        return data_t, target_t, event_t, length, idx
    

# -----------------------------------------------------------------------------
#        Some collate functions
# ----------------------------------------------------------------------------
def collate_fn(batch):
    """
    Provides mechanism to collate the batch

    ref: https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
    Puts data, and lengths into a packed_padded_sequence then returns
    the packed_padded_sequence and the labels.

    Parameters
    ----------
    batch: (list of tuples) [(*data, target)].
         data: all the differnt data input from `__getattr__`
         target: target y

    Returns
    -------
    packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
    target: (Tensor).

    """
    pad = C.PAD

    if len(batch) == 1:
        dx_t, dy_t, de_t, lengths, idx = batch[0]
        #  sigs = sigs.t()
        dx_t.unsqueeze_(0)
        dy_t.unsqueeze_(0)
        de_t.unsqueeze_(0)

        lengths = [lengths]
        idx = np.atleast_2d(idx)

    else:
        dx_t, dy_t, de_t, lengths, idx = zip(*[(dx, dy, de, length, idx)
                                               for (dx, dy, de, length, idx) 
                                               in sorted(batch, key=lambda x: x[3],
                                                         reverse=True)])
        max_len, n_feats = dx_t[0].size()
        device = dx_t[0].device

        dx_t = [T.cat((s, T.empty(max_len - s.size(0), n_feats, device=device).fill_(pad)), 0)
                if s.size(0) != max_len else s
                for s in dx_t]
        dx_t = T.stack(dx_t, 0).to(device)  # bs * max_seq_len * n_feat

        dy_t = T.stack(dy_t, 0).to(device)  # bs * n_out
        de_t = T.stack(de_t, 0).to(device)  # bs * n_out

        # Handling the other variables
        lengths = list(lengths)
        idx = np.vstack(idx) # bs * 1
    return dx_t, dy_t, de_t, lengths, idx
