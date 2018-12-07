# -*- coding: utf-8 -*-
'''
make figure showing observed times vs predicted times.

based on analysis in /tessorbitaldecay/src/verify_time_stamps.py
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, argparse, pickle, h5py

from astrobase.timeutils import get_epochs_given_midtimes_and_period
from numpy import array as nparr
from parse import parse, search

from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from scipy.stats import norm

import matplotlib.transforms as transforms

def linear_model(xdata, m, b):
    return m*xdata + b

def calculate_timing_accuracy(plname, period_guess):

    # load in the data with ONLY the literature times. fit a linear
    # ephemeris to it.
    manual_fpath = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/data/'+
        '{:s}_manual.csv'.format(plname)
    )
    mandf = pd.read_csv(manual_fpath, sep=';', comment=None)

    tmid = nparr(mandf['t0_BJD_TDB'])
    err_tmid = nparr(mandf['err_t0'])
    sel = np.isfinite(tmid) & np.isfinite(err_tmid)

    tmid = tmid[sel]
    err_tmid = err_tmid[sel]

    epoch, init_t0 = (
        get_epochs_given_midtimes_and_period(tmid, period_guess, verbose=True)
    )

    xdata = epoch
    ydata = tmid
    sigma = err_tmid

    popt, pcov = curve_fit(
        linear_model, xdata, ydata, p0=(period_guess, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_period_err = pcov[0,0]**0.5
    lsfit_t0 = popt[1]
    lsfit_t0_err = pcov[1,1]**0.5

    # now get observed tess times, and compare to predicted.
    sel_fpath = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/data/'+
        '{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'.
        format(plname)
    )
    seldf = pd.read_csv(sel_fpath, sep=';', comment=None)

    mytesstimes = nparr(seldf['original_reference'] == 'me')

    tess_tmid = nparr(seldf['sel_transit_times_BJD_TDB'])[mytesstimes]
    tess_err_tmid = nparr(seldf['err_sel_transit_times_BJD_TDB'])[mytesstimes]

    tess_sel = np.isfinite(tess_tmid) & np.isfinite(tess_err_tmid)
    if plname=='WASP-18b':
        tess_sel &= (tess_err_tmid*24*60 < 1)
    tess_tmid = tess_tmid[tess_sel]
    tess_err_tmid = tess_err_tmid[tess_sel]

    tess_epoch, _ = (
        get_epochs_given_midtimes_and_period(
            tess_tmid, period_guess, t0_fixed=lsfit_t0, verbose=True)
    )

    # now: calculate the uncertainty on the ephemeris during the time window that
    # tess observes, based on the literature values.
    tmid_expected = lsfit_t0 + lsfit_period*tess_epoch
    tmid_lower = (
        (lsfit_t0-lsfit_t0_err) +
        (lsfit_period-lsfit_period_err)*tess_epoch
    )
    tmid_upper = (
        (lsfit_t0+lsfit_t0_err) +
        (lsfit_period+lsfit_period_err)*tess_epoch
    )

    tmid_perr = (tmid_upper - tmid_expected)
    tmid_merr = (tmid_expected - tmid_lower)

    # difference between observed TESS time and expectation, in seconds
    diff_seconds = (tess_tmid - tmid_expected)*24*60*60
    err_prediction_seconds = np.mean([tmid_perr, tmid_merr], axis=0)*24*60*60

    return (
        lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
        epoch, tmid, err_tmid,
        tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
        err_prediction_seconds
    )




def plot_hjs():

    fig = plt.figure(figsize=(6,4))

    ax0 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax1 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax3 = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,3), colspan=2)
    axs = [ax0,ax1,ax2,ax3,ax4]

    plnames=['WASP-4b','WASP-5b','WASP-6b','WASP-18b','WASP-46b']
    periodguesses = [1.33823204, 1.6284246, 3.36100208, 0.94145299, 1.4303700]
    manualkdebandwidths = [1.5, 0.5, 1.5, 0.5, 0.8]

    for plname, ax, P_guess, bw in zip(
        plnames, axs, periodguesses, manualkdebandwidths
    ):

        # get difference between observed TESS time and expectation, in seconds
        (lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
        epoch, tmid, err_tmid,
        tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
        err_prediction_seconds ) = (
            calculate_timing_accuracy(
                plname=plname, period_guess=P_guess)
        )

        x = diff_seconds

        if plname=='WASP-4b':
            binned_tess_measurement_sigma = 5.20 # cf /src/plot_O_minus_C.py output
            prediction_precision = 9.4 # seconds, cf. elsewhere in this figure
            quoted_err =  np.sqrt(prediction_precision**2  +
                                  binned_tess_measurement_sigma**2)

            print('\n')
            print('*'*42)
            print('WASP-4b transits arrived {:.2f} +/- {:.2f} seconds early'.format(
                np.average(x, weights=1/err_prediction_seconds**2), quoted_err
            ))
            print('*'*42)
            print('\n')

        bw = bw*np.mean(err_prediction_seconds)

        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(x[:, None])

        # score_samples returns the log of the probability density
        meanerr = np.mean(err_prediction_seconds)
        x_d = np.linspace(-20*meanerr, 20*meanerr, num=1000)
        logprob = kde.score_samples(x_d[:, None])

        ax.plot(x_d, np.exp(logprob), color='#1f77b4', zorder=6,
                lw=0.5)
        l1 = ax.fill_between(x_d, np.exp(logprob), alpha=0.3,
                             label='KDE from TESS times',
                             color='#1f77b4', zorder=4, linewidth=0)

        # x coords are data, y coords are axes
        trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
        l2 = ax.scatter(x, np.full_like(x, 0.033), marker='|',
                        color='#1f77b4',
                        alpha=1,
                        label='observed TESS times', zorder=7,
                        linewidth=0.2, transform=trans)

        ax.plot(
            x_d, norm.pdf(x_d, loc=0, scale=meanerr),
            color='#ff7f0e',
            zorder=5, lw=0.5
        )
        l3 = ax.fill_between(
            x_d, norm.pdf(x_d, loc=0, scale=meanerr),
            color='#ff7f0e', alpha=0.3,
            label='predicted transit time'.
            format(meanerr),
            zorder=3, linewidth=0
        )

        print(plname, meanerr)
        sigtxt = (
            '$\sigma_{\mathrm{predicted}}$: '+
            '{:.1f} sec'.format(meanerr)
        )
        txt = '{:s}\n{:s}'.format(plname, sigtxt)
        if plname=='WASP-5b':
            xpos = -0.05
        else:
            xpos = 0
        ax.text(xpos, 0.98, txt,
                transform=ax.transAxes, color='black', fontsize=6,
                va='top', ha='left')

        ax.set_xlim([np.mean(x)-10*np.std(x), np.mean(x)+10*np.std(x)])

    fig.legend((l1,l2,l3),
               ('KDE from TESS', 'observed TESS',
                'literature prediction'),
               loc='upper right',
               bbox_to_anchor=(0.98, 0.42),
               fontsize='xx-small')

    for ax in axs:
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ylim = ax.get_ylim()
        ax.set_ylim((0, max(ylim)))
        ax.set_xlim((-310,310))

    fig.text(0.5,0, 'Deviation from predicted transit time [seconds]', ha='center')
    fig.text(0,0.5, 'Relative fraction', va='center', rotation=90)

    fig.tight_layout(h_pad=0, w_pad=0)

    for savname in ['../results/hjs.png', '../paper/f6.png',
                    '../results/hjs.pdf', '../paper/f6.pdf']:
       fig.savefig(savname, dpi=400, bbox_inches='tight')
       print('saved {:s}'.format(savname))

if __name__=="__main__":
    plot_hjs()
