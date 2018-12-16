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
    tmid_upper = np.maximum(
        (lsfit_t0+lsfit_t0_err) + tess_epoch*(lsfit_period+lsfit_period_err),
        (lsfit_t0+lsfit_t0_err) + tess_epoch*(lsfit_period-lsfit_period_err)
    )
    tmid_lower = np.minimum(
        (lsfit_t0-lsfit_t0_err) + tess_epoch*(lsfit_period-lsfit_period_err),
        (lsfit_t0-lsfit_t0_err) + tess_epoch*(lsfit_period+lsfit_period_err)
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

    for plname, ax, P_guess in zip(
        plnames, axs, periodguesses
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

        mean = np.average(x, weights=1/tess_err_tmid**2)
        std_of_OmC = np.std(x)
        uncert_in_mean = std_of_OmC / (len(x)-1)**(1/2)
        x_d = np.linspace(-20*std_of_OmC, 20*std_of_OmC, num=1000)

        # plot binned tess time distribution.
        ax.plot(x_d, norm.pdf(x_d, loc=mean, scale=uncert_in_mean),
                color='#1f77b4', zorder=6, lw=0.5)
        l1 = ax.fill_between(
            x_d, norm.pdf(x_d, loc=mean, scale=uncert_in_mean), alpha=0.3,
            label='binned TESS time', color='#1f77b4', zorder=4, linewidth=0
        )

        # plot the observed ticks. x coords are data, y coords are axes
        trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
        l2 = ax.scatter(x, np.full_like(x, 0.033), marker='|',
                        color='#1f77b4',
                        alpha=1,
                        label='observed TESS times', zorder=7,
                        linewidth=0.2, transform=trans)

        # plot predicted distribution.
        ax.plot(
            x_d, norm.pdf(x_d, loc=0, scale=np.mean(err_prediction_seconds)),
            color='#ff7f0e',
            zorder=5, lw=0.5
        )
        l3 = ax.fill_between(
            x_d, norm.pdf(x_d, loc=0, scale=np.mean(err_prediction_seconds)),
            color='#ff7f0e', alpha=0.3,
            label='predicted transit time'.
            format(std_of_OmC),
            zorder=3, linewidth=0
        )

        print(plname, std_of_OmC)
        # hard-coded unicode hyphen insanity
        sigtxt = (
            r'$\sigma_{\mathrm{pre'+u"\u2010"+'\! TESS}}$: '+
            '{:.1f} s'.format(np.mean(err_prediction_seconds))
        )
        muobstxt = '$\mu_{\mathrm{TESS}}$: '+'{:.1f} s'.format(mean)
        sigobstxt = '$\sigma_{\mathrm{TESS}}$: '+'{:.1f} s'.format(uncert_in_mean)
        txt = (
            '{:s}\n{:s}\n{:s}\n{:s}'.format(
                plname, sigtxt, muobstxt, sigobstxt)
        )
        if plname=='WASP-5b':
            xpos = -0.05
        else:
            xpos = 0
        ax.text(xpos, 0.98, txt,
                transform=ax.transAxes, color='black', fontsize=6,
                va='top', ha='left')

        ax.set_xlim([np.mean(x)-10*np.std(x), np.mean(x)+10*np.std(x)])

    fig.legend((l1,l2,l3),
               ('TESS binned', 'TESS individual',
                'pre-TESS prediction'),
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

    for savname in ['../results/hjs.png', '../paper/f7.png',
                    '../results/hjs.pdf', '../paper/f7.pdf']:
       fig.savefig(savname, dpi=500, bbox_inches='tight')
       print('saved {:s}'.format(savname))

if __name__=="__main__":
    plot_hjs()
