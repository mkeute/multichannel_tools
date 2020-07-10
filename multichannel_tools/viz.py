# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:11:51 2020

@author: neuro
"""
import numpy as np
from pylab import figure, show

def multichannel_plot(t, datarray):    

    fig = figure()

    

    yprops = dict(rotation=0,

                  horizontalalignment='right',

                  verticalalignment='center')

    

    axprops = dict(yticks=[])

    nchans = np.shape(datarray)[1]

    ht = .9/nchans

    ypos = [0.7, 0.5, 0.3, 0.1]

    ypos = np.linspace(1-ht, .1, nchans)

    

    for ch in range(nchans):

        ax1 =fig.add_axes([0.1, ypos[ch], 0.8, ht], **axprops)

        ax1.plot(t, datarray[:,ch])

        # ax1.set_ylabel(chan_labels[ch], **yprops)

        

        axprops['sharex'] = ax1

        axprops['sharey'] = ax1

    ax1.set_xlabel('time [s]')

    show()
    
    return fig, ax1



from mne.viz import plot_topomap
import numpy as np
import matplotlib.pyplot as plt


def get_channel_pos(channel_labels):
    from mne.channels import read_layout

    layout = read_layout("EEG1005")
    return (
        np.asanyarray([layout.pos[layout.names.index(ch)] for ch in channel_labels])[
            :, 0:2
        ]
        - 0.5
    ) * 2



def my_topomap(dat,chans, mask=None):

    pos = get_channel_pos(chans)
    im=plot_topomap(dat,pos, mask = mask)
    plt.colorbar(im[0])
    
    
    
def plot_with_errband(x,Y,col,effective_n=None):
    """

    Parameters
    ----------
    x : array
        vector of time stamps.
    Y : array
        time by units (e.g., subjects).

    Returns
    -------
    None.

    """
    from matplotlib import pyplot as plt
    import numpy as np
    if effective_n == None:
        np.shape(Y)[1] #effective_n can be used to account for repeatudes measures
    error = np.std(Y,axis=1)/np.sqrt(effective_n)
    y = np.mean(Y,axis = 1)
    
    plt.plot(x, y, 'k')
    f = plt.fill_between(x, y-error, y+error,facecolor=col, alpha=.5, edgecolor = None)
    plt.show()
    
    return f