# -*- coding: utf-8 -*-
'''
make future ephemeris prediction figure
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle

import emcee
from astropy.time import Time

from numpy import array as nparr

from plot_O_minus_C import get_data, linear_fit, quadratic_fit, precession_fit

def plot_future(
    x, y, sigma_y,
    x_occ, y_occ, sigma_y_occ,
    theta_linear, theta_quadratic, theta_prec,
    theta_linear_samples, theta_quad_samples, theta_prec_samples,
    refs,
    tcol, tcol_occ,
    savpath=None,
    xlabel='epoch', ylabel='deviation from constant period [min]',
    xlim=None, ylim=None, ylim1=None):

    xfit = np.linspace(10*np.min(x), 10*np.max(x), 10000)
    xfit_occ = np.linspace(10*np.min(x), 10*np.max(x), 10000)

    fig, (a0,a1) = plt.subplots(nrows=2, ncols=1, figsize=(6,4),
                                sharex=True)

    # converting epoch to time. first get tmid in days, then fix offset.
    tmid = (theta_linear[0] + x*theta_linear[1])/(24*60)
    t0_offset = int(tcol.split('_')[-2])
    tmid += t0_offset
    tmid = Time(tmid, format='jd', scale='tdb')

    tocc = (theta_linear[0] + x_occ*theta_linear[1])/(24*60)
    t0_offset = int(tcol_occ.split('_')[-2])
    tocc += t0_offset
    tocc = Time(tocc, format='jd', scale='tdb')

    tfit = (theta_linear[0] + xfit*theta_linear[1])/(24*60)
    t0_offset = int(tcol.split('_')[-2])
    tfit += t0_offset
    tfit = Time(tfit, format='jd', scale='tdb')

    ################
    # transit axis #
    ################
    is_tess = (refs=='me')

    for _x, _y, _err, _tmid in zip(
        x[~is_tess],y[~is_tess],sigma_y[~is_tess],tmid[~is_tess]
    ):

        a0.errorbar(_tmid.decimalyear,
                    nparr(_y-linear_fit(theta_linear, _x)),
                    _err,
                    fmt='.k', ecolor='black', zorder=2, mew=0,
                    ms=7,
                    elinewidth=1,
                    alpha= 1-(_err/np.max(sigma_y))**(1/2) + 0.1
                   )

    # bin TESS points &/or make a subplot
    tess_x = x[is_tess]
    tess_y = y[is_tess]
    tess_sigma_y = sigma_y[is_tess]

    bin_tess_y = np.average(nparr(tess_y-linear_fit(theta_linear, tess_x)),
                            weights=1/tess_sigma_y**2)
    bin_tess_sigma_y = np.mean(tess_sigma_y)/len(tess_y)**(1/2)
    bin_tess_x = np.median(tmid.decimalyear[is_tess])
    a0.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
            zorder=8, label='binned TESS time', markerfacecolor='yellow',
            markersize=8, marker='*', color='black', lw=0)

    # best-fit models
    a0.plot(tfit.decimalyear,
            quadratic_fit(theta_quadratic, xfit)
                - linear_fit(theta_linear, xfit),
            label='orbital decay', zorder=-1, c='#1f77b4')
    a0.plot(tfit.decimalyear,
            precession_fit(theta_prec, xfit)
                - linear_fit(theta_linear, xfit),
            label='apsidal precession', zorder=0, c='#ff7f0e')
    a0.plot(tfit.decimalyear,
            linear_fit(theta_linear, xfit)
                - linear_fit(theta_linear, xfit),
            label='constant period', zorder=-3, color='gray')

    for theta_quad_sample in theta_quad_samples:
        a0.plot(tfit.decimalyear,
                quadratic_fit(theta_quad_sample, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-2, alpha=0.17, c='#1f77b4')
    for theta_prec_sample in theta_prec_samples:
        a0.plot(tfit.decimalyear,
                precession_fit(theta_prec_sample, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-3, alpha=0.17, c='#ff7f0e')

    ################
    # occultations #
    ################
    a1.errorbar(tocc.decimalyear,
                y_occ-linear_fit(theta_linear, x, x_occ=x_occ)[1],
                sigma_y_occ, fmt='.k', ecolor='black', zorder=1, alpha=1,
                mew=1, elinewidth=1)
    # best-fit models
    a1.plot(tfit.decimalyear,
            quadratic_fit(theta_quadratic, xfit, x_occ=xfit_occ)[1]
                - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
            label='best quadratic fit', zorder=-1, c='#1f77b4')
    a1.plot(tfit.decimalyear,
            precession_fit(theta_prec, xfit, x_occ=xfit_occ)[1]
                - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
            label='best precession fit', zorder=0, c='#ff7f0e')
    a1.plot(tfit.decimalyear,
            linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1]
                - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
            label='best linear fit', zorder=-3, color='gray')

    for theta_quad_sample in theta_quad_samples:
        a1.plot(tfit.decimalyear,
                quadratic_fit(theta_quad_sample, xfit, x_occ=xfit_occ)[1]
                    - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
                zorder=-2, alpha=0.17, c='#1f77b4')
    for theta_prec_sample in theta_prec_samples:
        a1.plot(tfit.decimalyear,
                precession_fit(theta_prec_sample, xfit, x_occ=xfit_occ)[1]
                    - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
                zorder=-3, alpha=0.17, c='#ff7f0e')

    a0.legend(loc='lower left', fontsize='x-small')
    for ax in (a0,a1):
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.set_xlim(xlim)
    a0.set_ylim(ylim)
    a1.set_ylim(ylim1)

    fig.text(0.5,0, xlabel, ha='center')
    fig.text(0,0.5, ylabel, va='center', rotation=90)

    fig.tight_layout(h_pad=0, w_pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


def main(plname, xlim=None, ylim=None, savname=None, ylim1=None):

    basedir = '/home/luke/Dropbox/proj/tessorbitaldecay/'
    pkldir = basedir+'results/model_comparison/'+plname+'/'
    sampledir = '/home/luke/local/emcee_chains/'
    transitpath = (
        basedir+'data/{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'
        .format(plname)
    )
    occpath = (
        basedir+'data/{:s}_occultation_times_selected.csv'
        .format(plname)
    )

    linear_samplepath = os.path.join(
        sampledir,
        '{:s}_degree1_polynomial_timing_fit.h5'.format(plname)
    )
    quad_samplepath = os.path.join(
        sampledir,
        '{:s}_degree2_polynomial_timing_fit.h5'.format(plname)
    )
    prec_samplepath = os.path.join(
        sampledir,
        '{:s}_precession_timing_fit.h5'.format(plname)
    )
    model_samplepaths = [linear_samplepath,
                         quad_samplepath,
                         prec_samplepath]

    print('getting data from {:s}'.format(transitpath))
    x, y, sigma_y, data, tcol, refs = get_data(datacsv=transitpath)
    print('getting data from {:s}'.format(occpath))
    x_occ, y_occ, sigma_y_occ, data_occ, tcol_occ, refs_occ = (
        get_data(datacsv=occpath, is_occultation=True)
    )

    # get theta_linear, theta_quadratic from MCMC fits.
    fit_2d = pickle.load(open(pkldir+"fit_2d.pkl", 'rb'))
    fit_3d = pickle.load(open(pkldir+"fit_3d.pkl", 'rb'))
    fit_prec = pickle.load(open(pkldir+"fit_precession.pkl", 'rb'))

    medianparams_2d = fit_2d['fitinfo']['medianparams']
    medianparams_3d = fit_3d['fitinfo']['medianparams']
    medianparams_prec = fit_prec['fitinfo']['medianparams']
    best_theta_linear = nparr(
        [medianparams_2d['t0 [min]'], medianparams_2d['P [min]']]
    )
    best_theta_quadratic = nparr(
        [medianparams_3d['t0 [min]'],
         medianparams_3d['P [min]'],
         medianparams_3d['0.5 dP/dE [min]']
        ]
    )
    best_theta_prec = nparr(
        [medianparams_prec['t0 [min]'],
         medianparams_prec['P_side [min]'],
         medianparams_prec['e'],
         medianparams_prec['omega0'],
         medianparams_prec['domega_dE']
        ]
    )

    # get 100 random samples for each model 
    for ix, samplepath in enumerate(model_samplepaths):

        burninpercent = 0.3
        reader = emcee.backends.HDFBackend(samplepath)

        n_steps_performed = reader.iteration
        n_to_discard = int(burninpercent*n_steps_performed)

        samples = reader.get_chain(discard=n_to_discard, flat=True)

        n_samples = samples.shape[0]
        n_dim = samples.shape[1]

        n_desired = 100
        inds = np.random.choice(n_samples, n_desired, replace=False)

        sel_samples = samples[inds, :]

        if ix==0:
            theta_linear_samples = sel_samples
        elif ix==1:
            theta_quad_samples = sel_samples
        elif ix==2:
            theta_prec_samples = sel_samples


    if savname:
        savpath = os.path.join(pkldir, savname)
    else:
        savpath = os.path.join(pkldir, 'future.png')

    plot_future(
        x, y, sigma_y,
        x_occ, y_occ, sigma_y_occ,
        best_theta_linear, best_theta_quadratic, best_theta_prec,
        theta_linear_samples, theta_quad_samples, theta_prec_samples,
        refs,
        tcol, tcol_occ,
        savpath=savpath,
        xlabel='Gregorian year',
        ylabel='Deviation from constant period [minutes]',
        xlim=xlim, ylim=ylim, ylim1=ylim1)

    if savpath == os.path.join(pkldir, 'future.png'):
        copyfile(savpath, '../paper/f5.png')
        print('saved ../paper/f5.png')
        copyfile(savpath.replace('.png','.pdf'), '../paper/f5.pdf')
        print('saved ../paper/f5.pdf')


if __name__=="__main__":

    np.random.seed(42)

    #FIXME maybe better to argparse this...
    ticid = 402026209
    plname = 'WASP-4b'
    xlim = [2005, 2030]
    ylim = [-10.5,2]
    ylim1 = [-10.5,4.2]

    savname = None #'future_2005_to_2020.png'

    main(plname, xlim=xlim, ylim=ylim, savname=savname, ylim1=ylim1)
