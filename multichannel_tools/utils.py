
import numpy as np
import math
from scipy import signal
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