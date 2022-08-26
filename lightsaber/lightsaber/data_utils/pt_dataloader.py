#!/usr/bin/env python3

from typing import Optional, Any
from torch.utils.data import Dataset, DataLoader
from lightsaber import constants as C
from lightsaber.data_utils.datasets import EmptyDataset

import warnings


class PTDataloader(object):
    def __init__(self,
                 train_dataset: Optional[Dataset]=None,
                 val_dataset: Optional[Dataset]=None,
                 test_dataset: Optional[Dataset]=None,
                 cal_dataset: Optional[Dataset]=None,
                 pin_memory: Optional[str]='auto',
                 num_workers=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.cal_dataset = cal_dataset
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    # --------------------------------------------------------------
    #  Dataset handling section
    # -------------------------------------------------------------
    def _pin_memory(self):
        if self.pin_memory == 'auto':
            pin_memory = False
            try:
                if self.trainer.on_gpu:
                    pin_memory=True
            except AttributeError:
                pass
        else:
            pin_memory = self.pin_memory
        return pin_memory

    def train_dataloader(self):
        if self.val_dataset is None:
            warnings.warn('no dataset found. empty dataloader initialized', stacklevel=2)
            dataset = EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
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
        if self.val_dataset is None:
            warnings.warn('no dataset found. empty dataloader initialized', stacklevel=2)
            dataset = EmptyDataset()
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
        if self.test_dataset is None:
            warnings.warn('no dataset found. empty dataloader initialized', stacklevel=2)
            dataset = EmptyDataset()
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
        if self.cal_dataset is None:
            warnings.warn('no dataset found. empty dataloader initialized', stacklevel=2)
            dataset = EmptyDataset()
            dataloader = DataLoader(dataset)
        else:
            dataset = self.cal_dataset
            pin_memory = self._pin_memory()
            dataloader = DataLoader(self.cal_dataset, collate_fn=self.collate_fn, pin_memory=pin_memory,
                                    batch_size=self.hparams.batch_size,num_workers=self.num_workers)
        return dataloader
