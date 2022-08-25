#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:31:08 2022

@author: marius
"""


import numpy as np
from scipy import stats
import lmfit
from tqdm import tqdm
from scipy.stats import vonmises
from lmfit import Model
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS
from multiprocessing import Pool
from utils import cosine, do_one_perm



def cosine_fit(X, Y, stat = None, fixtures = {'v': 0, 'f': 1}, 
               parallel = True):
    """
    fit a cosine function of the form Y = a*(f*X+phi) + v,where a is the amplitude, 
    f is the normalized frequency, v is the vertical offset.
    Y is the input data (should have zero-mean and unit variance for interpretability),
    and X are corresponding phases in radians [0, 2*pi].
    Inputs:
    =======
        X (array-like, (N,)): phases in radians
        Y (array-like, (N,)): corresponding output values (should be standardized)
        stat (string): can be 'perm' (perform permutation test) or 'boot' (perform bootstrap).
                        Default (None): Return confidence bounds from lmfit
        fixtures: which parameters should be fixed. By default, v is fixed to 0 and f is fixed to 1,
            i.e., we estimate a single-period cosine fit on mean-centered data.
        parallel (bool): Should computations be parallelized?
    """
    
    
    
    #list all possible params and their plausible ranges, then extract
    #non-fixed parameters
    all_params = {'a':(0,np.ptp(Y)/2), 
                  'phi':(-np.pi, np.pi),
                  'f':(0.1, 100),
                  'v': (-100, 100)}
    
    model = lmfit.Model(cosine)
    params = model.make_params()
    for par,vals in all_params.items():
        if par in fixtures.keys():
            params[par].set(value=fixtures[par], vary = False)
        else:
            params[par].set(value=np.mean(vals), min=vals[0], max=vals[1])
    fit = model.fit(Y, params, x=X)


    if stat == 'perm':
        nperm = 5000
        dataperm = []
        
        for perm in range(nperm):
            dataperm.append([model,params, Y, np.random.permutation(X)])
        with Pool(8) as p:
            surrogate = p.starmap(do_one_perm, dataperm)
        
        stats = {'sig_thresh_5percent': np.percentile(surrogate, .95)}            
    elif stat == 'boot': 
        nsamp = 5000
        databoot = []
        
        for samp in range(nsamp):
            sampix = np.random.choice(range(len(X)), len(X), replace=True)
            databoot.append([model,params, Y[sampix], X[sampix]])
        with Pool(8) as p:
            bootdist = p.starmap(do_one_perm, dataperm)   
        stats = {'confidence_95_lower': np.percentile(bootdist, .025),
                 'confidence_95_upper': np.percentile(bootdist, .975)}
    elif stat == None:
        stats = {'confidence_95_lower': fit.conf_interval()[1],
                 'confidence_95_upper': fit.conf_interval()[-2]}
    return fit






if __name__ == '__main__':
    X=np.arange(0,2*np.pi,.01)
    #todo add parameters
    noiselevel = 2
    Y=stats.zscore(np.cos(X) + noiselevel*np.random.rand(len(X)))
    
    fit = cosine_fit(X, Y)
    plt.scatter(X,Y)
    plt.plot(X,fit.best_fit, 'r')
