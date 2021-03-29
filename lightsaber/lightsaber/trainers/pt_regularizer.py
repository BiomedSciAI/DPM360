#! /usr/bin/python 
"""
Library of regularizations

Some exeternal references: https://github.com/czifan/DeepSurv.pytorch/blob/master/networks.py#L12
"""

def l1_regularization(parameters, l1_strength):
    l1_reg = sum(param.abs().sum() for param in parameters)
    l1_loss =  l1_strength * l1_reg
    return l1_loss

def l2_regularization(parameters, l2_strength):
    l2_reg = sum(param.pow(2).sum() for param in parameters)
    l2_loss = l2_strength * l2_reg
    return l2_loss