# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:13:53 2020

@author: neuro
"""

import liesl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
sinfo = liesl.get_streaminfos_matching(type = 'EEG')
bfr = liesl.RingBuffer(sinfo[0], duration_in_ms = 30000)
bfr.start()
bfr.await_running()
fig = plt.figure()
t=bfr.get_data()
lines = []
notch = sg.butter(4,(49,51),btype='stop', fs=bfr.fs)
bandpass = sg.butter(4,(1,35),btype='pass', fs=bfr.fs)



while plt.fignum_exists(1):
    t=1e6*bfr.get_data()
    t=sg.lfilter(*bandpass,sg.lfilter(*notch,t,axis=0),axis=0)
    plt.clf()
    for ch in range(8):
        ax = plt.subplot(8,1,ch+1)
        ax.set_ylim((-50,50))
        tm = np.linspace(0,30,num=np.shape(t)[0])
        ax.plot(tm, t[:,ch])
        # lines.append(h)
    plt.pause(.1)



# while plt.fignum_exists(1):
#     for ch in range(8):
#         t=1e6*bfr.get_data()

#         h = lines[ch][0]
#         h.set_ydata(t[:,ch])
#     plt.pause(.1)
