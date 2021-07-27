#! /usr/bin/python 
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
