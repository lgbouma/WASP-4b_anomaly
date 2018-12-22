# -*- coding: utf-8 -*-
'''
make phase-folded lightcurve figure
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
import os, pickle
from shutil import copyfile

from astrobase.lcmath import phase_magseries
from astrobase.lcmath import phase_bin_magseries

def plot_phasefold(picklepath, xlim=None, ylim0=None, ylim1=None):

    # d['magseries'].keys()
    # dict_keys(['times', 'mags', 'errs', 'magsarefluxes'])
    # d['fitinfo'].keys()
    # dict_keys(['initialparams', 'initialmags', 'fixedparams',
    #            'finalparams', 'finalparamerrs', 'fitmags', 'fitepoch'])
    # d['fitinfo']['finalparams'].keys()
    # dict_keys(['incl', 'period', 'rp', 'sma', 't0', 'u_linear', 'u_quad'])
    d = pickle.load(open(picklepath, 'rb'))

    times = d['magseries']['times']
    fluxs = d['magseries']['mags']
    fitfluxs = d['fitinfo']['fitmags']

    fit_t0 = d['fitinfo']['finalparams']['t0']
    # NOTE: might want to use the best-fit value from the tables, instead of
    # the BLS period here.
    fit_period = d['fitinfo']['finalparams']['period']

    phzd = phase_magseries(times, fluxs, fit_period, fit_t0,
                           wrap=True, sort=True)
    fit_phzd = phase_magseries(times, fitfluxs, fit_period, fit_t0, wrap=True,
                               sort=True)

    phase = phzd['phase']
    phz_flux = phzd['mags']
    fit_phz_flux = fit_phzd['mags']

    binsize = 0.003
    bin_phzd = phase_bin_magseries(phase, phz_flux, binsize=binsize)
    bin_phase = bin_phzd['binnedphases']
    bin_fluxs = bin_phzd['binnedmags']
    fit_bin_phzd = phase_bin_magseries(phase, fit_phz_flux, binsize=binsize)
    fit_bin_phase = fit_bin_phzd['binnedphases']
    fit_bin_fluxs = fit_bin_phzd['binnedmags']

    ##########################################
    plt.close('all')
    f, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                               figsize=(0.8*6,0.8*4), gridspec_kw=
                               {'height_ratios':[3, 1]})

    a0.scatter(phase*fit_period*24, phz_flux, c='k', alpha=0.12, label='data',
               zorder=1, s=10, rasterized=True, linewidths=0)
    a0.plot(bin_phase*fit_period*24, bin_fluxs, alpha=1, mew=0.5,
            zorder=8, label='binned', markerfacecolor='yellow',
            markersize=8, marker='.', color='black', lw=0, rasterized=True)
    a0.plot(
        phase*fit_period*24, fit_phz_flux, c='#4346ff',
        zorder=2, rasterized=True, lw=1.5, alpha=0.7,
        label='model'
    )

    a1.scatter(
        phase*fit_period*24, phz_flux-fit_phz_flux, c='k', alpha=0.12,
        rasterized=True, s=10, linewidths=0, zorder=1
    )
    a1.plot(bin_phase*fit_period*24, bin_fluxs-fit_bin_fluxs, alpha=1, mew=0.5,
            zorder=8, markerfacecolor='yellow', markersize=8,
            marker='.', color='black', lw=0, rasterized=True)
    a1.plot(
        phase*fit_period*24, fit_phz_flux-fit_phz_flux, c='#4346ff',
        zorder=2, rasterized=True, lw=1.5, alpha=0.7
    )

    a1.set_xlabel('Time from mid-transit [hours]')
    a0.set_ylabel('Relative flux')
    a1.set_ylabel('Residual')
    for a in [a0, a1]:
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')
        if xlim:
            a.set_xlim(xlim)
    if ylim0:
        a0.set_ylim(ylim0)
    if ylim1:
        a1.set_ylim(ylim1)
    a0.legend(loc='best', fontsize='small')

    f.tight_layout(h_pad=-.3, w_pad=0)
    savpath = '../results/phasefold.png'
    f.savefig(savpath, dpi=600, bbox_inches='tight')
    print('made {}'.format(savpath))
    copyfile(savpath, '../paper/f2.png')
    print('saved ../paper/f2.png')


if __name__=="__main__":

    xlim = [-0.15*24,0.15*24]
    ylim0 = [0.96,1.013]
    ylim1 = [-0.018,0.018]

    pickledir = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/results/'
        'tess_lightcurve_fit_parameters/402026209/sector_2/'
    )
    picklename = '402026209_phased_mandelagol_fit_empiricalerrs.pickle'
    picklepath = os.path.join(pickledir, picklename)

    plot_phasefold(picklepath, xlim=xlim, ylim0=ylim0, ylim1=ylim1)
