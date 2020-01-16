# -*- coding: utf-8 -*-
"""
plot RV vs time, and residual (with best fit line model)
"""
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle

from astropy.time import Time
from astropy import units as units, constants as const

from numpy import array as nparr

import radvel
from radvel.plot import orbit_plots, mcmc_plots
from radvel.mcmc import statevars
from radvel.driver import save_status, load_status
import configparser

##########################################
# stuff needed to hack at radvel objects

class args_object(object):
    """
    a minimal version of the "parser" object that lets you work with the
    high-level radvel API from python. (without directly using the command line
    interface)
    """
    def __init__(self, setupfn, outputdir):
        # return args object with the following parameters set
        self.setupfn = setupfn
        self.outputdir = outputdir
        self.decorr = False
        self.plotkw = {}
        self.gp = False

# end of radvel hacks
##########################################

def _get_fit_results(setupfn, outputdir):

    args = args_object(setupfn, outputdir)
    args.inputdir = outputdir

    # radvel plot -t rv -s $basepath
    args.type = ['rv']

    # get residuals, RVs, error bars, etc from the fit that has been run..
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.inputdir, "{}_radvel.stat".format(conf_base))

    status = load_status(statfile)

    if not status.getboolean('fit', 'run'):
        raise AssertionError("Must perform max-liklihood fit before plotting")

    # initialize posterior object from the statfile that is passed.
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    # update the posterior to match the median best-fit parameters.
    summarycsv = os.path.join(outputdir, "WASP4_post_summary.csv")
    sdf = pd.read_csv(summarycsv)
    for param in [c for c in sdf.columns if 'Unnamed' not in c]:
        post.params[param] = radvel.Parameter(value=sdf.ix[1][param])

    P, _ = radvel.utils.initialize_posterior(config_file)
    if hasattr(P, 'bjd0'):
        args.plotkw['epoch'] = P.bjd0

    model = post.likelihood.model
    rvtimes = post.likelihood.x
    rvs = post.likelihood.y
    rverrs = post.likelihood.errorbars()
    num_planets = model.num_planets
    telvec = post.likelihood.telvec

    dvdt_merr = sdf['dvdt'].iloc[0]
    dvdt_perr = sdf['dvdt'].iloc[2]

    rawresid = post.likelihood.residuals()

    resid = (
        rawresid + post.params['dvdt'].value*(rvtimes-model.time_base)
        + post.params['curv'].value*(rvtimes-model.time_base)**2
    )

    rvtimes, rvs, rverrs, resid, telvec = rvtimes, rvs, rverrs, resid, telvec
    dvdt, curv = post.params['dvdt'].value, post.params['curv'].value
    dvdt_merr, dvdt_perr = dvdt_merr, dvdt_perr
    time_base = model.time_base

    return (rvtimes, rvs, rverrs, resid, telvec, dvdt, curv, dvdt_merr,
            dvdt_perr, time_base)


def main(make_my_plot=1):
    """
    args:

        hack_radvel_plots: debugging utility

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
    Pdot_tra = -4e-10
    Pdot_tra_perr = -4e-10 + 0.4e-10
    Pdot_tra_merr = -4e-10 - 0.4e-10
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
