# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:53:37 2020

@author: neuro
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.collections as collections
import numpy as np

def visual_inspection(x):
    x = np.array(x)
    x = x.flatten()
    nanix = np.zeros(len(x))
    def line_select_callback(eclick, erelease):
        """
        Callback for line selection.
    
        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    

    fig, current_ax = plt.subplots()                 # make a new plotting range
    # plt.plot(np.arange(len(x)), x, lw=1, c='b', alpha=.7)  # plot something
    current_ax.plot(x, 'b.', alpha=.7)  # plot something

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    RS = RectangleSelector(current_ax, line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
    RSinv = RectangleSelector(current_ax, line_select_callback,
                                drawtype='box', useblit=True,
                                button=[3],  # don't use middle button
                                minspanx=5, minspany=5,
                                spancoords='pixels',
                                interactive=True)
    plt.connect('key_press_event', (RS, RSinv))

    while plt.fignum_exists(1):
        plt.cla()
        current_ax.set_ylim([np.min(x[np.where(nanix == 0)[0]]), 1.1*np.max(x[np.where(nanix == 0)[0]])])
        current_ax.plot(x, 'b.', alpha=.7)  # plot something
        if np.sum(nanix) > 0:
            current_ax.plot(np.squeeze(np.where(nanix == 1)), x[np.where(nanix == 1)], 'w.', alpha=.7)  # plot something

        fig.show()
        plt.pause(.1)
        if plt.fignum_exists(1):
            plt.waitforbuttonpress(timeout = 2)
            
            if (RS.geometry[1][1] > 1):
                exclix = np.where((x > min(RS.geometry[0])) & (x < max(RS.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RS.geometry[1])) & (exclix < max(RS.geometry[1])))]
                nanix[exclix] = 1
            if (RSinv.geometry[1][1] > 1):
                exclix = np.where((x > min(RSinv.geometry[0])) & (x < max(RSinv.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RSinv.geometry[1])) & (exclix < max(RSinv.geometry[1])))]
                nanix[exclix] = 0
            if not plt.fignum_exists(1):
                break
            else:
                plt.pause(.1)
        else:
            plt.pause(.1)
            break
   
    return np.where(nanix == 0)[0]