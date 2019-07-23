# -*- coding: utf-8 -*-
'''
make O-C model figure
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle

from numpy import array as nparr

from astropy.time import Time
from astropy import units as units, constants as const

def get_data(
    datacsv='../data/WASP-18b_transits_and_TESS_times_O-C_vs_epoch_selected.csv',
    is_occultation=False
    ):
    # need to run make_parameter_vs_epoch_plots.py first; this generates the
    # SELECTED epochs (x values), mid-times (y values), and mid-time errors
    # (sigma_y).

    df = pd.read_csv(datacsv, sep=';')

    tcol = [c for c in nparr(df.columns) if
            ('times_BJD_TDB_minus_' in c) and ('minutes' in c)]
    if len(tcol) != 1:
        raise AssertionError('unexpected input file')
    else:
        tcol = tcol[0]

    if is_occultation:
        err_column = 'err_sel_occ_times_BJD_TDB_minutes'
    else:
        err_column = 'err_sel_transit_times_BJD_TDB_minutes'

    data = nparr( [
        nparr(df['sel_epoch']),
        nparr(df[tcol]),
        nparr(df[err_column])
    ])

    x, y, sigma_y = data
    refs = nparr(df['original_reference'])

    return x, y, sigma_y, data, tcol, refs


def linear_fit(theta, x, x_occ=None):
    """
    Linear model. Parameters (t0, P).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, period = theta
    if not isinstance(x_occ,np.ndarray):
        return t0 + period*x
    else:
        return t0 + period*x, t0 + period/2 + period*x_occ

def quadratic_fit(theta, x, x_occ=None):
    """
    Quadratic model. Parameters (t0, P, 0.5dP/dE).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, period, half_dP_dE = theta
    if not isinstance(x_occ,np.ndarray):
        return t0 + period*x + half_dP_dE*x**2
    else:
        return (t0 + period*x + half_dP_dE*x**2,
                t0 + period/2 + period*x_occ + half_dP_dE*x_occ**2
               )

def precession_fit(theta, x, x_occ=None):
    """
    Precession model. Parameters (t0, P_s, e, ω0, dω_by_dE).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, P_s, e, ω0, dω_by_dΕ = theta
    P_a = P_s * (1 - dω_by_dΕ/(2*np.pi))**(-1)

    if not isinstance(x_occ,np.ndarray):
        return (
            t0 + P_s*x -
            ( (e/np.pi) * P_a *
              np.cos( ω0 + dω_by_dΕ*x)
            )
        )
    else:
        return (
            t0 + P_s*x -
            ( (e/np.pi) * P_a *
              np.cos( ω0 + dω_by_dΕ*x)
            ),
            t0 + P_a/2 + P_s*x_occ +
            ( (e/np.pi) * P_a *
              np.cos( ω0 + dω_by_dΕ*x_occ)
            )
        )

def precession_fit_7param(theta, x, x_occ=None, Mstar=0.864, Mplanet=1.186):
    """
    Precession model. Parameters (t0, P_s, e, ω0, k2p, Rp, semimaj).
    Must pass transit times.

    If x_occ is none, returns model t_tra array.
    If x_occ is a numpy array, returns tuple of model t_tra and t_occ arrays.
    """
    t0, P_s, e, omega0, k2p, Rplanet, semimaj = theta

    domega_by_dΕ = (
        15*np.pi * k2p * ((Rplanet*units.Rjup/(semimaj*units.AU)).cgs.value)**5
        * (Mstar*units.Msun/(Mplanet*units.Mjup)).cgs.value
    )

    P_a = P_s * (1 - domega_by_dΕ/(2*np.pi))**(-1)

    if not isinstance(x_occ,np.ndarray):
        return (
            t0 + P_s*x -
            ( (e/np.pi) * P_a *
              np.cos( omega0 + domega_by_dΕ*x)
            )
        )
    else:
        return (
            t0 + P_s*x -
            ( (e/np.pi) * P_a *
              np.cos( omega0 + domega_by_dΕ*x)
            ),
            t0 + P_a/2 + P_s*x_occ +
            ( (e/np.pi) * P_a *
              np.cos( omega0 + domega_by_dΕ*x_occ)
            )
        )


def plot_O_minus_C(
    x, y, sigma_y, theta_linear, theta_quadratic, theta_prec,
    refs,
    savpath=None,
    xlabel='Epoch', ylabel='Deviation from constant period [min]',
    xlim=None, ylim=None, ylim1=None,
    include_all_points=False,
    x_extra=None, y_extra=None, sigma_y_extra=None,
    x_occ=None, y_occ=None, sigma_y_occ=None,
    onlytransits=False):

    xfit = np.linspace(10*np.min(x), 10*np.max(x), 10000)

    if onlytransits:

        fig, a0 = plt.subplots(nrows=1, ncols=1, figsize=(4*1.3,3*1.3))

        print('USING HACK TO NOT PLOT TESS DATA POINTS')
        istess = x > 1600

        # transit axis
        for e, tm, err in zip(x[~istess],y[~istess],sigma_y[~istess]):
            a0.errorbar(e,
                        nparr(tm-linear_fit(theta_linear, e)),
                        err,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=7,
                        elinewidth=1,
                        alpha= 1-(err/np.max(sigma_y))**(1/2) + 0.1
                       )

        # for legend
        a0.errorbar(9001, 9001, np.mean(err),
                    fmt='.k', ecolor='black', zorder=9, alpha=1, mew=1, ms=3,
                    elinewidth=1, label='pre-TESS')


        # bin TESS points
        is_tess = (refs=='me')
        tess_x = x[is_tess]
        tess_y = y[is_tess]
        tess_sigma_y = sigma_y[is_tess]

        bin_tess_y = np.average(nparr(tess_y-linear_fit(theta_linear, tess_x)),
                                weights=1/tess_sigma_y**2)
        bin_tess_sigma_y = np.mean(tess_sigma_y)/len(tess_y)**(1/2)
        bin_tess_x = np.median(tess_x)

        print('\n----- error on binned tess measurement -----\n')
        print('{:.2f} seconds'.format(bin_tess_sigma_y*60))

        #a0.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
        #        zorder=42, label='binned TESS', markerfacecolor='yellow',
        #        markersize=9, marker='*', color='black', lw=0)
        a0.errorbar(bin_tess_x, bin_tess_y, bin_tess_sigma_y,
                    alpha=1, zorder=11, label='binned TESS',
                    fmt='s', mfc='firebrick', elinewidth=1,
                    ms=3,
                    mec='firebrick',mew=1,
                    ecolor='firebrick')


        if include_all_points:
            a0.errorbar(x_extra,
                        nparr(y_extra-linear_fit(theta_linear, x_extra)),
                        sigma_y_extra,
                        fmt='.', color='g', ecolor='g', zorder=1,
                        alpha=1, mew=1, elinewidth=1)
            a0.text(0.96,0.02,
                    'Bixel+ 2018 IMACS\n'+
                    '(overplot; not included in fits)',
                   transform=a0.transAxes, color='g', fontsize='x-small',
                    va='bottom', ha='right')
            #a0.text(0.96,0.02,
            #        'select ETD times, & epoch from W+08\n'+
            #        '(overplot; not included in fits)',
            #       transform=a0.transAxes, color='g', fontsize='x-small',
            #        va='bottom', ha='right')

            # binned ETD point
            etd_bin = (x_extra > 0)
            etd_x = x_extra[etd_bin]
            etd_y = y_extra[etd_bin]
            etd_sigma_y = sigma_y_extra[etd_bin]

            bin_etd_y = np.average(
                nparr(etd_y-linear_fit(theta_linear, etd_x)),
                weights = 1/etd_sigma_y**2
            )
            bin_etd_sigma_y = np.std(etd_sigma_y)
            bin_etd_x = np.median(etd_x)

            # NOTE: sometimes might want to show!
            #a0.errorbar(bin_etd_x,
            #            bin_etd_y,
            #            bin_etd_sigma_y,
            #            fmt='.', color='red', ecolor='red',
            #            alpha=1, mew=1, elinewidth=1, zorder=8,
            #            label='binned ETD time')

        a0.plot(xfit,
                quadratic_fit(theta_quadratic, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-1)
        a0.plot(xfit,
                precession_fit(theta_prec, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-2)
        a0.plot(xfit,
                linear_fit(theta_linear, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-3, color='gray')

        # now move on to the occultation axis!
        #a0.text(0.98,0.95, 'Transits', transform=a0.transAxes, color='k',
        #        fontsize='medium', va='top', ha='right')

        # add "time" axis on top
        # make twin axis to show year on top
        period = 1.338231466*units.day
        t0 = 2455804.515752*units.day
        transittimes = x*period + t0
        times = Time(transittimes, format='jd', scale='tdb')
        a_top = a0.twiny()
        a_top.scatter(times.decimalyear, np.zeros_like(times), s=0)
        a_top.set_xlabel('Year', fontsize='large')

        # hidden point for a1 legend
        #a1.plot(1500, 3, alpha=1, mew=0.5,
        #        zorder=-3, label='binned TESS time', markerfacecolor='yellow',
        #        markersize=9, marker='*', color='black', lw=0)

        if not include_all_points:
            a0.legend(loc=(0.5, 0.02), fontsize='x-small', framealpha=1)
        else:
            a0.legend(loc=(0.2, 0.02), fontsize='x-small', framealpha=1)

        a0.get_yaxis().set_tick_params(which='both', direction='in')
        a0.get_xaxis().set_tick_params(which='both', direction='in')
        a0.set_xlim(xlim)
        a_top.get_yaxis().set_tick_params(which='both', direction='in')
        a_top.get_xaxis().set_tick_params(which='both', direction='in')
        a0.set_ylim((-1.6, 1.6))

        fig.text(0.5,0, xlabel, ha='center', fontsize='large')
        fig.text(-0.02,0.5, 'Transit obs. - calc. [minutes]', va='center', rotation=90, fontsize='large')

        fig.tight_layout(h_pad=0, w_pad=0)
        savpath = savpath.replace('.png','_onlytransits.png')
        fig.savefig(savpath, bbox_inches='tight', dpi=400)
        print('made {:s}'.format(savpath))
        savpath = savpath.replace('.png','_onlytransits.pdf')
        fig.savefig(savpath, bbox_inches='tight')
        print('made {:s}'.format(savpath))



    elif (
        not onlytransits
        and
        isinstance(x_occ,np.ndarray)
        and
        isinstance(y_occ,np.ndarray)
        and
        isinstance(sigma_y_occ,np.ndarray)
    ):
        xfit_occ = np.linspace(10*np.min(x), 10*np.max(x), 10000)

        fig, (a0,a1) = plt.subplots(nrows=2, ncols=1, figsize=(6,4),
                                    sharex=True)

        print('USING HACK TO NOT PLOT TESS DATA POINTS')
        istess = x > 1600

        # transit axis
        for e, tm, err in zip(x[~istess],y[~istess],sigma_y[~istess]):
            a0.errorbar(e,
                        nparr(tm-linear_fit(theta_linear, e)),
                        err,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=7,
                        elinewidth=1,
                        alpha= 1-(err/np.max(sigma_y))**(1/2) + 0.1
                       )

        # for legend
        a0.errorbar(9001, 9001, np.mean(err),
                    fmt='.k', ecolor='black', zorder=9, alpha=1, mew=1, ms=3,
                    elinewidth=1, label='pre-TESS')


        # bin TESS points
        is_tess = (refs=='me')
        tess_x = x[is_tess]
        tess_y = y[is_tess]
        tess_sigma_y = sigma_y[is_tess]

        bin_tess_y = np.average(nparr(tess_y-linear_fit(theta_linear, tess_x)),
                                weights=1/tess_sigma_y**2)
        bin_tess_sigma_y = np.mean(tess_sigma_y)/len(tess_y)**(1/2)
        bin_tess_x = np.median(tess_x)

        print('\n----- error on binned tess measurement -----\n')
        print('{:.2f} seconds'.format(bin_tess_sigma_y*60))

        #a0.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
        #        zorder=42, label='binned TESS', markerfacecolor='yellow',
        #        markersize=9, marker='*', color='black', lw=0)
        a0.errorbar(bin_tess_x, bin_tess_y, bin_tess_sigma_y,
                    alpha=1, zorder=11, label='binned TESS',
                    fmt='s', mfc='firebrick', elinewidth=1,
                    ms=3,
                    mec='firebrick',mew=1,
                    ecolor='firebrick')


        if include_all_points:
            a0.errorbar(x_extra,
                        nparr(y_extra-linear_fit(theta_linear, x_extra)),
                        sigma_y_extra,
                        fmt='.', color='g', ecolor='g', zorder=1,
                        alpha=1, mew=1, elinewidth=1)
            a0.text(0.96,0.02,
                    'Bixel+ 2018 IMACS\n'+
                    '(overplot; not included in fits)',
                   transform=a0.transAxes, color='g', fontsize='x-small',
                    va='bottom', ha='right')
            #a0.text(0.96,0.02,
            #        'select ETD times, & epoch from W+08\n'+
            #        '(overplot; not included in fits)',
            #       transform=a0.transAxes, color='g', fontsize='x-small',
            #        va='bottom', ha='right')

            # binned ETD point
            etd_bin = (x_extra > 0)
            etd_x = x_extra[etd_bin]
            etd_y = y_extra[etd_bin]
            etd_sigma_y = sigma_y_extra[etd_bin]

            bin_etd_y = np.average(
                nparr(etd_y-linear_fit(theta_linear, etd_x)),
                weights = 1/etd_sigma_y**2
            )
            bin_etd_sigma_y = np.std(etd_sigma_y)
            bin_etd_x = np.median(etd_x)

            # NOTE: sometimes might want to show!
            #a0.errorbar(bin_etd_x,
            #            bin_etd_y,
            #            bin_etd_sigma_y,
            #            fmt='.', color='red', ecolor='red',
            #            alpha=1, mew=1, elinewidth=1, zorder=8,
            #            label='binned ETD time')

        a0.plot(xfit,
                quadratic_fit(theta_quadratic, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-1)
        a0.plot(xfit,
                precession_fit(theta_prec, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-2)
        a0.plot(xfit,
                linear_fit(theta_linear, xfit)
                    - linear_fit(theta_linear, xfit),
                zorder=-3, color='gray')

        # now move on to the occultation axis!
        a1.errorbar(x_occ,
                    y_occ-linear_fit(theta_linear, x, x_occ=x_occ)[1],
                    sigma_y_occ, fmt='.k', ecolor='black', zorder=2, alpha=1,
                    mew=1, elinewidth=1)
        a1.plot(xfit_occ,
                quadratic_fit(theta_quadratic, xfit, x_occ=xfit_occ)[1]
                   - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
                label='quadratic fit', zorder=-1)
        a1.plot(xfit_occ,
                precession_fit(theta_prec, xfit, x_occ=xfit_occ)[1]
                    - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
                label='precession fit', zorder=-1)
        a1.plot(xfit_occ,
                linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1]
                   - linear_fit(theta_linear, xfit, x_occ=xfit_occ)[1],
                label='linear fit', zorder=-3, color='gray')

        a0.text(0.98,0.95, 'Transits', transform=a0.transAxes, color='k',
                fontsize='medium', va='top', ha='right')
        a1.text(0.98,0.05, 'Occultations', transform=a1.transAxes, color='k',
                fontsize='medium', va='bottom', ha='right')

        # add "time" axis on top
        # make twin axis to show year on top
        period = 1.338231466*units.day
        t0 = 2455804.515752*units.day
        transittimes = x*period + t0
        times = Time(transittimes, format='jd', scale='tdb')
        a_top = a0.twiny()
        a_top.scatter(times.decimalyear, np.zeros_like(times), s=0)
        a_top.set_xlabel('Year', fontsize='large')

        # hidden point for a1 legend
        #a1.plot(1500, 3, alpha=1, mew=0.5,
        #        zorder=-3, label='binned TESS time', markerfacecolor='yellow',
        #        markersize=9, marker='*', color='black', lw=0)

        if not include_all_points:
            a0.legend(loc=(0.5, 0.02), fontsize='x-small', framealpha=1)
        else:
            a0.legend(loc=(0.2, 0.02), fontsize='x-small', framealpha=1)
        a1.legend(loc='upper right', fontsize='x-small', framealpha=1)
        for ax in (a0,a1):
            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')
            ax.set_xlim(xlim)
        a_top.get_yaxis().set_tick_params(which='both', direction='in')
        a_top.get_xaxis().set_tick_params(which='both', direction='in')
        a0.set_ylim(ylim)
        a1.set_ylim(ylim1)

        fig.text(0.5,0, xlabel, ha='center', fontsize='large')
        fig.text(-0.02,0.5, ylabel, va='center', rotation=90, fontsize='large')

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.savefig(savpath, bbox_inches='tight', dpi=400)
        print('made {:s}'.format(savpath))
        savpath = savpath.replace('.png','.pdf')
        fig.savefig(savpath, bbox_inches='tight')
        print('made {:s}'.format(savpath))


    else:
        fig, ax = plt.subplots(figsize=(6,4))

        cuterr = np.percentile(sigma_y, 50)
        print('showing points with err > {:.2f} seconds as solid'.
              format(cuterr*60))
        sel_solid = sigma_y <= cuterr
        sel_seethru = ~sel_solid

        # solid black
        ax.errorbar(x[sel_solid],
                    nparr(y-linear_fit(theta_linear, x))[sel_solid],
                    sigma_y[sel_solid],
                    fmt='.k', ecolor='black', zorder=2, alpha=1, mew=1,
                    elinewidth=1)

        # gray
        ax.errorbar(x[sel_seethru],
                    nparr(y-linear_fit(theta_linear, x))[sel_seethru],
                    sigma_y[sel_seethru],
                    fmt='.', color='lightgray', ecolor='lightgray', zorder=1,
                    alpha=1, mew=1, elinewidth=1)

        # bin TESS points &/or make a subplot
        is_tess = (refs=='me')
        tess_x = x[is_tess]
        tess_y = y[is_tess]
        tess_sigma_y = sigma_y[is_tess]

        bin_tess_y = np.average(nparr(tess_y-linear_fit(theta_linear, tess_x)),
                               weights=1/tess_sigma_y**2)
        bin_tess_sigma_y = np.mean(tess_sigma_y)/len(tess_y)**(1/2)
        bin_tess_x = np.median(tess_x)

        print('\n----- error on binned tess measurement -----\n')
        print('{:.2f} seconds'.format(bin_tess_sigma_y*60))

        ax.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
                zorder=8, label='binned TESS time', markerfacecolor='yellow',
                markersize=9, marker='*', color='black', lw=0)

        if include_all_points:
            ax.errorbar(x_extra,
                        nparr(y_extra-linear_fit(theta_linear, x_extra)),
                        sigma_y_extra,
                        fmt='.', color='g', ecolor='g', zorder=1,
                        alpha=1, mew=1, elinewidth=1)
            ax.text(0.96,0.02,
                    'select ETD times, & epoch from W+08\n'+
                    '(overplot; not included in fits)',
                   transform=ax.transAxes, color='g', fontsize='x-small',
                    va='bottom', ha='right')

            # binned ETD point
            etd_bin = (x_extra > 0)
            etd_x = x_extra[etd_bin]
            etd_y = y_extra[etd_bin]
            etd_sigma_y = sigma_y_extra[etd_bin]

            bin_etd_y = np.average(
                nparr(etd_y-linear_fit(theta_linear, etd_x)),
                weights = 1/etd_sigma_y**2
            )
            bin_etd_sigma_y = np.mean(etd_sigma_y)/len(etd_y)**(1/2)
            bin_etd_x = np.median(etd_x)

            ax.errorbar(bin_etd_x,
                        bin_etd_y,
                        bin_etd_sigma_y,
                        fmt='.', color='red', ecolor='red',
                        alpha=1, mew=1, elinewidth=1, zorder=8,
                        label='binned ETD time')

        ax.plot(xfit,
                quadratic_fit(theta_quadratic, xfit)
                    - linear_fit(theta_linear, xfit),
                label='best quadratic fit', zorder=-1)
        ax.plot(xfit,
                precession_fit(theta_prec, xfit)
                    - linear_fit(theta_linear, xfit),
                label='best precession fit', zorder=-2)
        ax.plot(xfit,
                linear_fit(theta_linear, xfit)
                    - linear_fit(theta_linear, xfit),
                label='best linear fit', zorder=-3, color='gray')

        ax.legend(loc='best', fontsize='x-small')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        fig.tight_layout(h_pad=0, w_pad=0)

        fig.tight_layout()
        fig.savefig(savpath, bbox_inches='tight', dpi=400)
        print('saved {:s}'.format(savpath))
        savpath = savpath.replace('.png','.pdf')
        fig.savefig(savpath, bbox_inches='tight')
        print('saved {:s}'.format(savpath))


def main(plname, xlim=None, ylim=None, include_all_points=False, ylim1=None):

    basedir = '/home/luke/Dropbox/proj/tessorbitaldecay/'
    transitpath = (
        basedir+'data/literature_plus_TESS_times/{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'
        .format(plname)
    )
    occpath = (
        basedir+'data/literature_plus_TESS_times/{:s}_occultation_times_selected.csv'
        .format(plname)
    )

    pkldir = basedir+'results/model_comparison/'+plname+'_arxiv_submitted/'

    print('getting data from {:s}'.format(transitpath))
    x, y, sigma_y, data, tcol, refs = get_data(datacsv=transitpath)
    print('getting data from {:s}'.format(occpath))
    x_occ, y_occ, sigma_y_occ, data_occ, occ_tcol, occ_refs = (
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

    if include_all_points:
        # retrieve extra points from csv file.
        extrafile = '../data/WASP-4b_manual_extra.csv'
        edf = pd.read_csv(extrafile, sep=';')
        t_mid_extra = nparr(edf['t0_BJD_TDB'])
        err_tmid_extra = nparr(edf['err_t0'])

        # tmid = t0 + epoch*period
        # epoch = (tmid-t0)/period
        t0, P_fit = medianparams_2d['t0 [min]'], medianparams_2d['P [min]']

        P_fit = P_fit/(60*24)
        t0 = t0/(60*24)
        t0_offset = int(tcol.split('_')[-2])
        t0 = t0 + t0_offset

        epoch = (t_mid_extra - t0) / P_fit
        epoch_int = np.round(epoch, 0)

        print('\nepoch before rounding\n')
        print(epoch)
        print('\nafter rounding\n')
        print(epoch_int)

        x_extra = epoch_int
        y_extra = (t_mid_extra-t0_offset)*24*60
        sigma_y_extra = err_tmid_extra*24*60
    else:
        x_extra, y_extra, sigma_y_extra = None, None, None


    if not include_all_points:
        savpath = pkldir + 'O_minus_C.png'
    else:
        savpath = pkldir + 'O_minus_C_includeallpoints.png'

    onlytransits = True
    plot_O_minus_C(
        x, y, sigma_y,
        best_theta_linear, best_theta_quadratic, best_theta_prec,
        refs,
        savpath=savpath,
        xlabel='Epoch',
        ylabel='Deviation from linear fit [minutes]',
        xlim=xlim, ylim=ylim,
        include_all_points=include_all_points,
        x_extra=x_extra, y_extra=y_extra, sigma_y_extra=sigma_y_extra,
        x_occ=x_occ, y_occ=y_occ, sigma_y_occ=sigma_y_occ, ylim1=ylim1,
        onlytransits=onlytransits)

    if not include_all_points and not onlytransits:
        copyfile(savpath, '../paper/f4.png')
        print('saved ../paper/f4.png')
        copyfile(savpath.replace('.png','.pdf'), '../paper/f4.pdf')
        print('saved ../paper/f4.pdf')
    elif include_all_points and not onlytransits:
        copyfile(savpath, '../paper/f9.png')
        print('saved ../paper/f9.png')
        copyfile(savpath.replace('.png','.pdf'), '../paper/f9.pdf')
        print('saved ../paper/f9.pdf')
    elif onlytransits:
        print('made the O-C plot with only transits')


if __name__=="__main__":

    ticid = 402026209
    plname = 'WASP-4b'

    # with selected points used in fit
    xlim = [-1100,2000]
    ylim = [-1.6,1.3]
    ylim1 = [-5,4.2]
    main(plname, xlim=xlim, ylim=ylim, include_all_points=False, ylim1=ylim1)

    # with all points
    ylim = [-1.6,1.3]
    xlim = [-1100,2000]
    ylim1 = [-5,4.2]
    main(plname, xlim=xlim, ylim=ylim, include_all_points=True, ylim1=ylim1)
