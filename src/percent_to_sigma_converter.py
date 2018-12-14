# -*- coding: utf-8 -*-
'''
something happens BLAH percent of the time. how many sigma is this?
'''
from __future__ import division, print_function

from astropy.table import Table

import matplotlib.pyplot as plt, pandas as pd, numpy as np

from scipy.interpolate import interp1d
from scipy.stats import norm

def get_sigma_given_percent(percent):

    x_grid = np.linspace(-10, 10, 50000)

    pdf = norm.pdf(x_grid)

    cdf = norm.cdf(x_grid)

    sigma_away = x_grid[np.argmin( np.abs( cdf - percent ) )]

    print('{:.8f} corresponds to {:.8f} sigma'.format(percent, sigma_away))


if __name__=="__main__":

    print('WASP-6b, t_systematic_offset < -81.50 sec ruled out at XX sigma')
    val_6b = 0.000000439560
    frac = 1 - val_6b
    get_sigma_given_percent(frac)

    print('WASP-46b, t_systematic_offset < -81.50 sec ruled out at XX sigma')
    val_46b = 0.022780460097
    frac = 1 - val_46b
    get_sigma_given_percent(frac)

    print('WASP-18b, t_systematic_offset < -81.50 sec ruled out at XX sigma')
    val_18b = 0.012510650383
    frac = 1- val_18b
    get_sigma_given_percent(frac)

    print('multiply 6b,18b, 46b: t_systematic_offset < -81.50 sec ruled out at XX sigma')
    frac = 1- val_6b*val_46b*val_18b
    print('product is {}'.format(val_6b*val_46b*val_18b))
    get_sigma_given_percent(frac)
