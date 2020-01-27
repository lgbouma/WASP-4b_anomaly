"""
plot RV vs time, and residual (with best fit line model)
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle

from numpy import array as nparr
from astropy.time import Time
from astropy import units as units, constants as const

from radvel_utils import args_object, _get_fit_results

def main(make_my_plot=1):
    """
    args:

        make_my_plot: assumes you have run the radvel fit. reads in the output
        values to plot.
    """

    # initialization script used to make the fix_gammadot fits
    basedir = os.path.join(
        os.path.expanduser('~'),
        "Dropbox/proj/WASP-4b_anomaly/"
    )

    setupfn = os.path.join(basedir,"src/WASP4.py")

    outputdir = os.path.join(
        basedir,
        "results/rv_fitting/LGB_20190911_fix_gammaddot"
    )
    savpath='../results/20190911_rv_fit.png'

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, outputdir
    )

    #
    # make the plot
    #
    offset=2450000

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    utel = np.unique(telvec)
    markers = ['o','s','^']
    for ix, tel in enumerate(utel):
        sel = (telvec == tel)

        ax.errorbar(rvtimes[sel]-offset, resid[sel], rverrs[sel], marker=markers[ix],
                    ecolor='gray', zorder=10, mew=0, ms=4, elinewidth=1,
                    color='C{}'.format(ix), lw=0, label=tel)

    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)

    assert curv==0
    model_line = dvdt*(_times-time_base)# + curv*(_times-time_base)**2

    model_merr = dvdt_merr*(_times-time_base)# + curv*(_times-time_base)**2
    model_perr = dvdt_perr*(_times-time_base)# + curv*(_times-time_base)**2

    ax.plot(_times-offset, model_line, color='black', zorder=-3, lw=0.5)
    ax.fill_between(_times-offset, model_merr, model_perr, color='black',
                    zorder=-4, alpha=0.2, lw=0)#label='$\pm 1\sigma$')
    ax.text(0.55, 0.54, 'Best-fit from RVs', va='bottom', ha='left',
            transform=ax.transAxes, color='black')

    # what would explain the Pdot from transits?
    period = 1.338231466*units.day
    Pdot_tra = -2.736e-10
    Pdot_tra_perr = Pdot_tra + 2.83e-11
    Pdot_tra_merr = Pdot_tra - 2.83e-11
    dvdt_tra = (Pdot_tra * const.c / period).to(
        (units.m/units.s)/units.day).value
    dvdt_tra_perr = (Pdot_tra_perr * const.c / period).to(
        (units.m/units.s)/units.day).value
    dvdt_tra_merr = (Pdot_tra_merr * const.c / period).to(
        (units.m/units.s)/units.day).value

    # model times are now an arrow band
    _mtimes = np.linspace(np.min(rvtimes)+500, np.min(rvtimes)+1500, num=2000)
    _mbase = np.nanmedian(_mtimes)
    model_tra_line = dvdt_tra*(_mtimes-_mbase)
    model_tra_merr = dvdt_tra_merr*(_mtimes-_mbase)# + curv*(_times-time_base)**2
    model_tra_perr = dvdt_tra_perr*(_mtimes-_mbase)# + curv*(_times-time_base)**2

    ax.plot(_mtimes-offset, model_tra_line-150,
            color='purple', zorder=-3, lw=0.5, ls=':')
    ax.fill_between(_mtimes-offset, model_tra_merr-150, model_tra_perr-150,
                    color='purple', zorder=-4, alpha=0.4, lw=0)
    ax.text(0.05, 0.12, 'Slope = $c\dot{P}/P$', va='bottom',
            ha='left', transform=ax.transAxes, color='purple', alpha=0.9)

    ax.legend(loc='upper right', fontsize='medium')

    ax.set_xlabel('JD'+' - {}'.format(offset), fontsize='large')
    ax.set_ylabel('RV obs. - calc. [m/s]', fontsize='large')

    # make twin axis to show year on top
    times = Time(rvtimes, format='jd', scale='tdb')
    a_top = ax.twiny()
    a_top.scatter(times.decimalyear, rvs, s=0)
    a_top.set_xlabel('Year', fontsize='large')

    ax.set_xlim((3950, 9050))
    ax.set_ylim((-300, 300))

    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)
    a_top.set_xlim(
        (Time( (3950+2450000), format='jd', scale='tdb').decimalyear,
        Time( (9050+2450000), format='jd', scale='tdb').decimalyear)
    )

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.tick_params(right=True, which='both', direction='in')
    a_top.get_yaxis().set_tick_params(which='both', direction='in')
    a_top.get_xaxis().set_tick_params(which='both', direction='in')

    fig.tight_layout(h_pad=0.15, w_pad=0, pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


if __name__=="__main__":

    make_my_plot = 1
    main(make_my_plot=make_my_plot)
