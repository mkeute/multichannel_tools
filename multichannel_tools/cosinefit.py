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
from functools import partial



def cosine(x, a, phi, f, v):
    return a * np.cos(f*x + phi) + v


def cosine_fit(X, Y, backend = 'scipy', stat = None, fixtures = {'v': 0, 'f': 1}, 
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
        backend (string): can be 'scipy' or 'fitlm'
        stat (string): can be 'perm' (perform permutation test) or 'boot' (perform bootstrap)
        fixtures: which parameters should be fixed. By default, v is fixed to 0 and f is fixed to 1,
            i.e., we estimate a single-period cosine fit on mean-centered data.
        parallel (bool): Should computations be parallelized?
    """
    
    
    from multiprocessing import Pool
    
    #list all possible params and their plausible ranges, then extract
    #non-fixed parameters
    all_params = {'a':(0,np.ptp(Y)/2), 
                  'phi':(-np.pi, np.pi),
                  'f':(0.1, 100),
                  'v': (-100, 100)}
    
    if backend == 'lmfit':
        model = lmfit.Model(cosine)
        params = model.make_params()
        for par,vals in all_params.items():
            if par in fixtures.keys():
                params[par].set(value=fixtures[par], vary = False)
            else:
                params[par].set(value=np.random.uniform(*vals), min=vals[0], max=vals[1])
        fit = model.fit(Y, params, x=X)
    elif backend == 'scipy':
        pass
        
        
        
        
        
        
        
        
        # if perm:
        #     model = lmfit.Model(sinus)
        #     params = result.params
        #     surrogate = np.nan*np.zeros(nperm)
        #     dataperm = []
        #     for perm in tqdm(range(nperm)):
        #         permdf = df.copy().loc[df.freq == frequency,]
        #         permdf['tarphase']=np.random.permutation(permdf['tarphase'])
        #         # for rec in permdf.recording.unique():
        #         #     permdf.loc[permdf.recording == rec,key] = np.random.permutation(permdf.loc[permdf.recording == rec,key])
        #         # permdf = normalize_by_rec(permdf,key=key,method=scaling, per_freq=True)

        #         y = permdf[scaling+key].to_numpy()
        #         x = permdf['tarphase'].to_numpy()
        #         keep = np.invert(np.isnan(y) | np.isinf(y))
        #         y = y[keep]
        #         x = x[keep]
        #         x = np.deg2rad(x)
        #         dataperm.append([model,params, y, x])
        #     with Pool(4) as p:
        #         surrogate = p.starmap(do_one_perm, dataperm)
        # else: 
        #     surrogate = [np.nan]
            
        
    return fit








if __name__ == '__main__':
    X=np.arange(0,2*np.pi,.01)
    #todo add parameters
    noiselevel = 2
    Y=stats.zscore(np.cos(X) + noiselevel*np.random.rand(len(X)))
    
    fit = cosine_fit(X, Y,backend='lmfit')
    plt.scatter(X,Y)
    plt.plot(X,fit.best_fit, 'r')
