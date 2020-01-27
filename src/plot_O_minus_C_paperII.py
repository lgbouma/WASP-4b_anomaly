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

def plot_O_minus_C(
    x, y, sigma_y, theta_linear, theta_quadratic, theta_prec,
    refs,
    savpath=None,
    xlabel='Epoch', ylabel='Deviation from constant period [min]',
    xlim=None, ylim=None, ylim1=None,
    include_all_points=False,
    x_extra=None, y_extra=None, sigma_y_extra=None,
    x_occ=None, y_occ=None, sigma_y_occ=None,
    onlytransits=False, theta_quad_merr=None, theta_quad_perr=None):

    xfit = np.linspace(10*np.min(x), 10*np.max(x), 10000)

    if not onlytransits:
        raise NotImplementedError

    fig, a0 = plt.subplots(nrows=1, ncols=1, figsize=(4*1.3,3*1.3))

    print('USING HACK TO NOT PLOT TESS DATA POINTS')
    istess = (x > 1624) & (x < 1645)

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

    a0.plot(bin_tess_x, bin_tess_y, alpha=1, mew=0.5,
            zorder=42, label='binned TESS', markerfacecolor='yellow',
            markersize=9, marker='*', color='black', lw=0)
    a0.errorbar(bin_tess_x, bin_tess_y, bin_tess_sigma_y,
                alpha=1, zorder=11, label='binned TESS',
                fmt='s', mfc='firebrick', elinewidth=1,
                ms=0,
                mec='black',mew=1,
                ecolor='black')


    if include_all_points:
        raise NotImplementedError

    a0.plot(xfit,
            quadratic_fit(theta_quadratic, xfit)
                - linear_fit(theta_linear, xfit),
            zorder=-1, color='C0')
    a0.plot(xfit,
            linear_fit(theta_linear, xfit)
                - linear_fit(theta_linear, xfit),
            zorder=-3, color='gray')

    a0.plot(xfit,
            quadratic_fit(theta_quad_merr, xfit)
                - linear_fit(theta_linear, xfit),
            zorder=-2, color='C0', alpha=0.5)
    a0.plot(xfit,
            quadratic_fit(theta_quad_perr, xfit)
                - linear_fit(theta_linear, xfit),
            zorder=-2, color='C0', alpha=0.5)
    a0.fill_between(
            xfit,
            quadratic_fit(theta_quad_merr, xfit)
                - linear_fit(theta_linear, xfit),
            quadratic_fit(theta_quad_perr, xfit)
                - linear_fit(theta_linear, xfit),
            zorder=-3, color='C0', alpha=0.2
    )


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

    #if not include_all_points:
    #    a0.legend(loc='upper right', fontsize='x-small', framealpha=1)
    #else:
    #    a0.legend(loc='upper right', fontsize='x-small', framealpha=1)

    a0.get_yaxis().set_tick_params(which='both', direction='in')
    a0.get_xaxis().set_tick_params(which='both', direction='in')
    a0.tick_params(right=True, which='both', direction='in')
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
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('made {:s}'.format(savpath))



def main(plname, xlim=None, ylim=None, include_all_points=False, ylim1=None):

    homedir = os.path.expanduser('~')
    basedir = os.path.join(homedir, 'Dropbox/proj/tessorbitaldecay/')
    transitpath = (
        basedir+'data/literature_plus_TESS_times/{:s}_literature_and_TESS_times_O-C_vs_epoch_20200127_selected.csv'
        .format(plname)
    )
    occpath = 'foo.csv'

    pkldir = basedir+'results/model_comparison/WASP-4b_20200127/'

    print('getting data from {:s}'.format(transitpath))
    x, y, sigma_y, data, tcol, refs = get_data(datacsv=transitpath)
    print('getting data from {:s}'.format(occpath))

    if os.path.exists(occpath):
        x_occ, y_occ, sigma_y_occ, data_occ, occ_tcol, occ_refs = (
            get_data(datacsv=occpath, is_occultation=True)
        )
    else:
        x_occ, y_occ, sigma_y_occ, data_occ, occ_tcol, occ_refs = (
            None, None, None, None, None, None
        )

    # get theta_linear, theta_quadratic from MCMC fits.
    fit_2d = pickle.load(open(pkldir+"fit_2d.pkl", 'rb'))
    fit_3d = pickle.load(open(pkldir+"fit_3d.pkl", 'rb'))

    medianparams_2d = fit_2d['fitinfo']['medianparams']
    medianparams_3d = fit_3d['fitinfo']['medianparams']
    merrs_3d = fit_3d['fitinfo']['std_merrs']
    perrs_3d = fit_3d['fitinfo']['std_perrs']
    best_theta_linear = nparr(
        [medianparams_2d['t0 [min]'], medianparams_2d['P [min]']]
    )
    best_theta_quadratic = nparr(
        [medianparams_3d['t0 [min]'],
         medianparams_3d['P [min]'],
         medianparams_3d['0.5 dP/dE [min]']
        ]
    )
    theta_quad_merr = nparr(
        [medianparams_3d['t0 [min]'],
         medianparams_3d['P [min]'],
         medianparams_3d['0.5 dP/dE [min]'] - merrs_3d['0.5 dP/dE [min]']
        ]
    )
    theta_quad_perr = nparr(
        [medianparams_3d['t0 [min]'],
         medianparams_3d['P [min]'],
         medianparams_3d['0.5 dP/dE [min]'] + perrs_3d['0.5 dP/dE [min]']
        ]
    )

    x_extra, y_extra, sigma_y_extra = None, None, None

    savpath = '../results/O_minus_C_20200127.png'

    onlytransits = True
    plot_O_minus_C(
        x, y, sigma_y,
        best_theta_linear, best_theta_quadratic, None,
        refs,
        savpath=savpath,
        xlabel='Epoch',
        ylabel='Deviation from linear fit [minutes]',
        xlim=xlim, ylim=ylim,
        include_all_points=include_all_points,
        x_extra=x_extra, y_extra=y_extra, sigma_y_extra=sigma_y_extra,
        x_occ=x_occ, y_occ=y_occ, sigma_y_occ=sigma_y_occ, ylim1=ylim1,
        onlytransits=onlytransits, theta_quad_merr=theta_quad_merr,
        theta_quad_perr=theta_quad_perr)

    if onlytransits:
        print('made the O-C plot with only transits')


if __name__=="__main__":

    ticid = 402026209
    plname = 'WASP-4b'

    # with all points
    ylim = [-5,5]
    xlim = [-1500, 2000]
    ylim1 = None #FIXME [-5,4.2]
    main(plname, xlim=xlim, ylim=ylim, ylim1=ylim1)
