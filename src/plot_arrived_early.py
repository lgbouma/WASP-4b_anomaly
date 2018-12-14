# -*- coding: utf-8 -*-
'''
make O-C plot that shows the TESS times arrived early.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle

from numpy import array as nparr

from plot_hjs import calculate_timing_accuracy

def linear_model(t0, period, x):
    return t0 + x*period

def plot_arrived_early(plname, xlim=None, ylim=None, savpath=None, ylim1=None):

    plname='WASP-4b'
    period_guess = 1.33823204

    # get difference between observed TESS time and expectation, in seconds
    (lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
    epoch, tmid, err_tmid,
    tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
    err_prediction_seconds ) = (
        calculate_timing_accuracy(
            plname=plname, period_guess=period_guess)
    )

    model_epoch = np.arange(-1000,3000,1)
    model_tmid = lsfit_t0 + model_epoch*lsfit_period

    model_tmid_upper = np.maximum(
        (lsfit_t0+lsfit_t0_err) + model_epoch*(lsfit_period+lsfit_period_err),
        (lsfit_t0+lsfit_t0_err) + model_epoch*(lsfit_period-lsfit_period_err)
    )
    model_tmid_lower = np.minimum(
        (lsfit_t0-lsfit_t0_err) + model_epoch*(lsfit_period-lsfit_period_err),
        (lsfit_t0-lsfit_t0_err) + model_epoch*(lsfit_period+lsfit_period_err)
    )

    # make the plot
    fig,(a0,a1) = plt.subplots(nrows=2,ncols=1,figsize=(4,7))

    # transit axis
    _ix = 0
    for e, tm, err in zip(epoch,tmid,err_tmid):
        if _ix == 0:
            a0.errorbar(e,
                        nparr(tm - linear_model(
                            lsfit_t0, lsfit_period, e))*24*60,
                        err*24*60,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=6,
                        elinewidth=1,
                        alpha= 1-(err/np.max(err_tmid))**(1/2) + 0.1,
                        label='pre-TESS')
            _ix += 1
        else:
            a0.errorbar(e,
                        nparr(tm - linear_model(
                            lsfit_t0, lsfit_period, e))*24*60,
                        err*24*60,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=7,
                        elinewidth=1,
                        alpha= 1-(err/np.max(err_tmid))**(1/2) + 0.1
                       )

    for ax in [a1]:
        ax.errorbar(tess_epoch,
                    nparr(tess_tmid -
                          linear_model(lsfit_t0, lsfit_period, tess_epoch))*24*60,
                    tess_err_tmid*24*60,
                    fmt='sk', ecolor='black', zorder=9, alpha=1, mew=1,
                    ms=3,
                    elinewidth=1,
                    label='TESS')

    # for the legend
    # a0.errorbar(9001, 9001, np.mean(tess_err_tmid*24*60),
    #             fmt='sk', ecolor='black', zorder=9, alpha=1, mew=1, ms=3,
    #             elinewidth=1, label='TESS')


    bin_tess_y = np.average(nparr(
        tess_tmid-linear_model(lsfit_t0, lsfit_period, tess_epoch)),
        weights=1/tess_err_tmid**2
    )
    bin_tess_err_tmid = np.mean(tess_err_tmid)/len(tess_tmid)**(1/2)

    tess_yval = nparr(tess_tmid -
                      linear_model(lsfit_t0, lsfit_period, tess_epoch))*24*60
    print('bin_tess_y (min)'.format(bin_tess_y))
    print('bin_tess_y (sec)'.format(bin_tess_y*60))
    print('std (min) {}'.format(np.std(tess_yval)))
    print('std (sec): {}'.format(np.std(tess_yval)*60))
    print('error measurement (plotted, min): {}'.format(bin_tess_err_tmid*24*60))
    print('error measurement (plotted, sec): {}'.format(bin_tess_err_tmid*24*60*60))
    bin_tess_x = np.median(tess_epoch)

    for ax in [a0]:
        ax.errorbar(bin_tess_x, bin_tess_y*24*60, bin_tess_err_tmid*24*60,
                    alpha=1, zorder=11, label='binned TESS',
                    fmt='s', mfc='firebrick', elinewidth=1,
                    ms=3,
                    mec='firebrick',mew=1,
                    ecolor='firebrick')

    yupper = (
        model_tmid_upper -
        linear_model(lsfit_t0, lsfit_period, model_epoch)
    )
    ylower = (
        model_tmid_lower -
        linear_model(lsfit_t0, lsfit_period, model_epoch)
    )
    err_pred = ( (
        yupper[np.argmin(np.abs(model_epoch-bin_tess_x))]
        -
        ylower[np.argmin(np.abs(model_epoch-bin_tess_x))])
        /2
    )

    print('error prediction (min): {}'.format(err_pred*24*60))
    print('error prediction (sec): {}'.format(err_pred*24*60*60))

    print('in abstract: arrived {:.2f} +/- {:.2f} sec early'.
          format(bin_tess_y*24*60*60, ((err_pred*24*60*60)**2 +
                                       (bin_tess_err_tmid*24*60*60)**2)**(1/2))
    )

    for ax in (a0,a1):
        ax.plot(model_epoch, yupper*24*60, color='#1f77b4', zorder=-1, lw=0.5)
        ax.plot(model_epoch, ylower*24*60, color='#1f77b4', zorder=-1, lw=0.5)
    a0.fill_between(model_epoch, ylower*24*60, yupper*24*60, alpha=0.3,
                    color='#1f77b4', zorder=-2, linewidth=0)
    a1.fill_between(model_epoch, ylower*24*60, yupper*24*60, alpha=0.3,
                    label='pre-TESS ephemeris', color='#1f77b4', zorder=-2,
                    linewidth=0)

    bin_yupper = np.ones_like(model_epoch)*( bin_tess_y*24*60 +
                                            bin_tess_err_tmid*24*60 )
    bin_ylower = np.ones_like(model_epoch)*( bin_tess_y*24*60 -
                                            bin_tess_err_tmid*24*60 )
    a1.plot(model_epoch, bin_yupper, color='firebrick', zorder=-1, lw=0.5)
    a1.plot(model_epoch, bin_ylower, color='firebrick', zorder=-1, lw=0.5)
    a1.fill_between(model_epoch, bin_ylower, bin_yupper, alpha=0.3,
                    color='firebrick', zorder=-2, linewidth=0,
                    label='binned TESS')

    a0.legend(loc='upper right', fontsize='small')
    a1.legend(loc='upper right', fontsize='small')
    a1.set_xlabel('Epoch')
    fig.text(0.,0.5, 'Deviation from predicted transit time [minutes]',
             va='center', rotation=90)
    if xlim:
        a0.set_xlim(xlim)
    if ylim:
        a0.set_ylim(ylim)
    if ylim1:
        a1.set_ylim(ylim1)
    ax.set_xlim((np.floor(bin_tess_x-1.1*len(tess_epoch)/2),
                 np.ceil(bin_tess_x+1.1*len(tess_epoch)/2)))

    a0.text(0.03,0.97,'All transits',ha='left',
            va='top',fontsize='medium',transform=a0.transAxes)
    a1.text(0.03,0.97,'TESS transits',ha='left',
            va='top',fontsize='medium',transform=a1.transAxes)

    for ax in (a0,a1):
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    fig.tight_layout(h_pad=0.05, w_pad=0)

    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


if __name__=="__main__":

    plname = 'WASP-4b'
    savpath = '../results/arrived_early.png'

    # with selected points used in fit
    xlim = [-600,2600]
    ylim = [-2.5,1.5]
    ylim1 = [-2.5,1.5] # for bottom subplot
    plot_arrived_early(plname, xlim=xlim, ylim=ylim, savpath=savpath,
                       ylim1=ylim1)

    copyfile(savpath, '../paper/f3.png')
    print('saved ../paper/f3.png')
    copyfile(savpath.replace('.png','.pdf'), '../paper/f3.pdf')
    print('saved ../paper/f3.pdf')

