'''
File: visualizations.py
File Created: 2021-11-03
Author: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk)
-----
Last Modified: 2021-11-03 10:57:13 pm
Modified By: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk>)
'''


import sklearn
import pandas as pd
import numpy as np
import numbers
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.special import expit



def discrete_time_plot(time_steps, predicted_probabilities, max_time, vertical_line_at = None):

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_time)

    ax.step(time_steps, predicted_probabilities)
    ax.margins(x=0.01)

    if vertical_line_at != None:
        plt.axvline(x=vertical_line_at, label='Death', color = 'black')
    
    plt.legend(loc=0)

    return fig, ax


def hazard_plot(time_steps, predicted_probabilities, max_time, vertical_line_at = None):

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_time)

    ax.plot(time_steps, predicted_probabilities)
    ax.margins(x=0.01)

    if vertical_line_at != None:
        plt.axvline(x=vertical_line_at, label='Death', color = 'black')
    
    plt.legend(loc=0)
    
    return fig, ax

