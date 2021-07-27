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

import numpy as np
from lightsaber import constants as C


from sklearn.linear_model import LogisticRegression
 
def logit_pvalue(x, yhat, coef_, intercept_):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters
    ----------
    x: matrix on which the model was fit
    yhat: predicted probabilities by applying model on X
    coef_: coefficient of fitted logistic regression model
    intercept_: intercept of fitted logistic regression model
    
    Returns
    -------
    p: pvalues for coefficients
    t: t statistic for coefficients
    se: standard error for coefficients
   
    Notes
    -----
    This function uses asymtptics for maximum likelihood estimates.
    
    Ref: https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance/47079198#47079198
    """
    # yhat = model.predict_proba(x)
    n = len(yhat)
    
    # m = len(model.coef_[0]) + 1
    # coefs = np.concatenate([model.intercept_, model.coef_[0]])
    m = len(coef_)
    coefs = np.concatenate([intercept_, coef_])
    
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p, t, se
