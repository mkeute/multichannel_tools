
import numpy as np
import math
from scipy import signal
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def get_PLV(x1,x2):
    """
    get phase-locking value of two 1d-arrays
    """
    sig1_hill=signal.hilbert(x1)
    sig2_hill=signal.hilbert(x2)
    theta1=np.unwrap(np.angle(sig1_hill))
    theta2=np.unwrap(np.angle(sig2_hill))
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv



def common_avg_ref(dat):
    """
    

    Parameters
    ----------
    dat : array(n_samples,n_electrodes)
        EEG data.

    Returns
    -------
    dat_reref. Same size array as dat, common avg. rereferenced.

    """
    
    bl = np.tile(np.nanmean(dat,axis = 1,keepdims = True), (1,dat.shape[1]))
    dat -= bl    
    return dat


def TKEO(x):
    y = x[1:-1]**2 - x[:-2]*x[2:]
    
    y = np.concatenate([[y[0]],y,[y[-1]]])
    
    return y
        






def regress_out_pupils(raw, ocular_channels = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'], method = 'PCA'):
    
    """
    raw: Continuous raw data in MNE format
    ocular_channels: can be labels of EOG channels or EEG channels close to the
        eyes if no EOG was recorded
    method: how to combine the ocular channels. Can be 'PCA', 'mean', or 'median'.
    """
    
    raw_data = raw.get_data(picks = 'eeg')
    ocular_data = raw.get_data(picks = ocular_channels)
    
    if method == 'PCA':
        pca = PCA()
        comps = pca.fit_transform(ocular_data.T)
        ocular_chan = comps[:,0]
    elif method == 'mean':
        ocular_chan = np.mean(ocular_data, axis = 0)
    elif method == 'median':
        ocular_chan = np.median(ocular_data, axis = 0)
    
    for ch in range(raw_data.shape[0]):
        m = OLS(raw_data[ch,:], ocular_chan)
        raw_data[ch,:] -= m.fit().predict()
    raw._data[:raw_data.shape[0],:] = raw_data
    return raw