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

def plot_arrived_early(plname, xlim=None, ylim=None, savpath=None):

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

    # model_tmid_upper = (
    #     (lsfit_t0+lsfit_t0_err) + model_epoch*(lsfit_period+lsfit_period_err)
    # )
    # model_tmid_lower = (
    #     (lsfit_t0-lsfit_t0_err) + model_epoch*(lsfit_period-lsfit_period_err)
    # )
    model_tmid_upper = (
        (lsfit_t0) + model_epoch*(lsfit_period+lsfit_period_err)
    )
    model_tmid_lower = (
        (lsfit_t0) + model_epoch*(lsfit_period-lsfit_period_err)
    )

    # make the plot
    fig,ax = plt.subplots(figsize=(4,4))

    # transit axis
    cuterr = np.percentile(err_tmid, 50)
    print('showing points with err > {:.2f} seconds as solid'.
          format(cuterr*60))
    sel_solid = (err_tmid <= cuterr) & (err_tmid*24*60 < 1)
    sel_seethru = (~sel_solid)  & (err_tmid*24*60 < 1)

    # solid black
    ax.errorbar(epoch[sel_solid],
                nparr(tmid - linear_model(
                    lsfit_t0, lsfit_period, epoch))[sel_solid]*24*60,
                err_tmid[sel_solid]*24*60,
                fmt='.k', ecolor='black', zorder=10, alpha=1, mew=1,
                elinewidth=1,
                label='$\sigma_{t_{\mathrm{tra}}} \geq \mathrm{median}(\sigma_{t_{\mathrm{tra}}})$')

    # gray
    ax.errorbar(epoch[sel_seethru],
                nparr(tmid -
                      linear_model(lsfit_t0, lsfit_period, epoch))[sel_seethru]*24*60,
                err_tmid[sel_seethru]*24*60,
                fmt='.', color='lightgray', ecolor='lightgray',
                zorder=8,
                alpha=1, mew=1, elinewidth=1,
                label='$\sigma_{t_{\mathrm{tra}}} < \mathrm{median}(\sigma_{t_{\mathrm{tra}}})$')

    ax.errorbar(tess_epoch,
                nparr(tess_tmid -
                      linear_model(lsfit_t0, lsfit_period, tess_epoch))*24*60,
                tess_err_tmid*24*60,
                fmt='sk', ecolor='black', zorder=9, alpha=1, mew=1,
                ms=3,
                elinewidth=1,
                label='TESS observation')

    bin_tess_y = np.average(nparr(
        tess_tmid-linear_model(lsfit_t0, lsfit_period, tess_epoch)),
        weights=1/tess_err_tmid**2
    )
    bin_tess_err_tmid = np.mean(tess_err_tmid)/len(tess_tmid)**(1/2)
    bin_tess_x = np.median(tess_epoch)

    ax.errorbar(bin_tess_x, bin_tess_y*24*60, bin_tess_err_tmid*24*60,
                alpha=1, zorder=11, label='binned TESS',
                fmt='s', mfc='red', elinewidth=1,
                ms=3,
                mec='red',mew=1,
                ecolor='red')

    yupper = (
        model_tmid_upper -
        linear_model(lsfit_t0, lsfit_period, model_epoch)
    )
    ylower = (
        model_tmid_lower -
        linear_model(lsfit_t0, lsfit_period, model_epoch)
    )

    ax.plot(model_epoch, yupper*24*60, color='#1f77b4', zorder=-1, lw=0.5)
    ax.plot(model_epoch, ylower*24*60, color='#1f77b4', zorder=-1, lw=0.5)
    l1 = ax.fill_between(model_epoch, ylower*24*60, yupper*24*60, alpha=0.3,
                         label='literature prediction',
                         color='#1f77b4', zorder=-2, linewidth=0)

    ax.legend(loc='best', fontsize='xx-small')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Deviation from predicted transit time [minutes]')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

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
    plot_arrived_early(plname, xlim=xlim, ylim=ylim, savpath=savpath)

    copyfile(savpath, '../paper/f2.png')
    print('saved ../paper/f2.png')
    copyfile(savpath.replace('.png','.pdf'), '../paper/f2.pdf')
    print('saved ../paper/f2.pdf')

