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
# a bunch of stuff needed to hack at radvel objects

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

def _hack_radvel_plots(args):
    """
    Hacked version of radvel.driver.plots

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(
        args.inputdir, "{}_radvel.stat".format(conf_base)
    )

    status = load_status(statfile)

    assert status.getboolean('fit', 'run'), \
        "Must perform max-liklihood fit before plotting"
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    # update the plotted posterior to match the median best-fit parameters.
    summarycsv = "../results/rv_fitting/t10onlytwo_LGB_20190228_fix_gammaddot/WASP4_post_summary.csv"
    sdf = pd.read_csv(summarycsv)
    for param in [c for c in sdf.columns if 'Unnamed' not in c]:
        post.params[param] = radvel.Parameter(value=sdf.ix[1][param])

    for ptype in args.type:
        print("Creating {} plot for {}".format(ptype, conf_base))

        if ptype == 'rv':
            args.plotkw['uparams'] = post.uparams
            saveto = os.path.join(
                args.outputdir,conf_base+'_rv_multipanel.pdf'
            )
            P, _ = radvel.utils.initialize_posterior(config_file)
            if hasattr(P, 'bjd0'):
                args.plotkw['epoch'] = P.bjd0

            #FIXME FIXME: see line 103 of this for how to access residuals
            model = post.likelihood.model
            rvtimes = post.likelihood.x
            rverr = post.likelihood.errorbars()
            num_planets = model.num_planets

            #FIXME FIXME
            rawresid = post.likelihood.residuals()

            resid = (
                rawresid + post.params['dvdt'].value*(rvtimes-model.time_base)
                + post.params['curv'].value*(rvtimes-model.time_base)**2
            )

            import IPython; IPython.embed()

            RVPlot = orbit_plots.MultipanelPlot(
                post, saveplot=saveto, **args.plotkw
            )
            RVPlot.plot_multipanel()

            # check to make sure that Posterior is not GP, print warning if it is
            if isinstance(post.likelihood, radvel.likelihood.CompositeLikelihood):
                like_list = post.likelihood.like_list
            else:
                like_list = [post.likelihood]
            for like in like_list:
                if isinstance(like, radvel.likelihood.GPLikelihood):
                    print("WARNING: GP Likelihood(s) detected."
                          "You may want to use the '--gp' flag"
                          "when making these plots.")
                    break

        else:
            raise NotImplementedError('nope')

        savestate = {'{}_plot'.format(ptype): os.path.relpath(saveto)}
        save_status(statfile, 'plot', savestate)

# end of radvel hacks
##########################################


def plot_rvs(
        rvtimes, rvs, rverrs, resid, telvec,
        dvdt, curv,
        dvdt_merr, dvdt_perr,
        time_base,
        savpath='../paper/f6.png',
        offset=2450000
):

    fig, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                               figsize=(0.8*6,0.8*5.5), gridspec_kw=
                               {'height_ratios':[3, 2]})

    utel = np.unique(telvec)
    markers = ['o','s','^']
    for ix, tel in enumerate(utel):
        sel = (telvec == tel)
        a0.errorbar(rvtimes[sel]-offset, rvs[sel], rverrs[sel], marker=markers[ix],
                    ecolor='gray', zorder=10, mew=0, ms=4, elinewidth=1,
                    color='C{}'.format(ix), label=tel, lw=0)

        a1.errorbar(rvtimes[sel]-offset, resid[sel], rverrs[sel], marker=markers[ix],
                    ecolor='gray', zorder=10, mew=0, ms=4, elinewidth=1,
                    color='C{}'.format(ix), lw=0)

    _times = np.linspace(np.min(rvtimes)-1000, np.max(rvtimes)+1000,
                              num=1000)
    assert curv==0
    model_line = dvdt*(_times-time_base)# + curv*(_times-time_base)**2

    model_merr = dvdt_merr*(_times-time_base)# + curv*(_times-time_base)**2
    model_perr = dvdt_perr*(_times-time_base)# + curv*(_times-time_base)**2

    a1.plot(_times-offset, model_line, label='Best-fit $\dot{v_r}$ from RVs',
            color='black', zorder=-3, lw=0.5)
    a1.fill_between(_times-offset, model_merr, model_perr, color='black',
                    zorder=-4, alpha=0.2, lw=0)#label='$\pm 1\sigma$')

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

    model_tra_line = dvdt_tra*(_times-time_base)
    model_tra_merr = dvdt_tra_merr*(_times-time_base)# + curv*(_times-time_base)**2
    model_tra_perr = dvdt_tra_perr*(_times-time_base)# + curv*(_times-time_base)**2

    a1.plot(_times-offset, model_tra_line, label='Required $\dot{v_r}$ from transits',
            color='C4', zorder=-3, lw=0.5, ls=':')
    a1.fill_between(_times-offset, model_tra_merr, model_tra_perr, color='C4',
                    zorder=-4, alpha=0.2, lw=0)#label='$\pm 1\sigma$')

    a0.legend(loc='upper center', fontsize='medium')
    a1.legend(loc='upper right', fontsize='medium')

    a0.set_ylabel('Radial velocity [m/s]', fontsize='large')
    a1.set_xlabel('JD'+' - {}'.format(offset),
                  fontsize='large')
    a1.set_ylabel('Residual [m/s]', fontsize='large')

    # make twin axis to show year on top
    times = Time(rvtimes, format='jd', scale='tdb')
    a_top = a0.twiny()
    a_top.scatter(times.decimalyear, rvs, s=0)
    a_top.set_xlabel('Year', fontsize='large')

    for a in [a0,a1]:
        a.set_xlim(np.min(rvtimes)-50-offset, np.max(rvtimes)+50-offset)
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')

    a_top.get_yaxis().set_tick_params(which='both', direction='in')
    a_top.get_xaxis().set_tick_params(which='both', direction='in')

    a0.set_ylim((-410,410))

    fig.tight_layout(h_pad=0.15, w_pad=0, pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


def main(hack_radvel_plots=0, make_my_plot=1):

    # initialization script used to make the fix_gammadot fits
    basedir = "/home/luke/Dropbox/proj/WASP-4b_decay/"
    setupfn = os.path.join(basedir,"src/WASP4.py")
    outputdir = os.path.join(
        basedir, "results/rv_fitting/fix_gammaddot_for_paper")

    args = args_object(setupfn, outputdir)
    args.inputdir = os.path.join(
        basedir,"results/rv_fitting/t10onlytwo_LGB_20190228_fix_gammaddot")

    # radvel plot -t rv -s $basepath
    args.type = ['rv']
    if hack_radvel_plots:
        _hack_radvel_plots(args)

    # get residuals, RVs, error bars, etc.
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(
        args.inputdir, "{}_radvel.stat".format(conf_base)
    )

    status = load_status(statfile)

    assert status.getboolean('fit', 'run'), \
        "Must perform max-liklihood fit before plotting"
    # initialize posterior object from the statfile that is passed.
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    # update the posterior to match the median best-fit parameters.
    summarycsv = "../results/rv_fitting/t10onlytwo_LGB_20190228_fix_gammaddot/WASP4_post_summary.csv"
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

    plot_rvs(
        rvtimes, rvs, rverrs, resid, telvec,
        post.params['dvdt'].value, post.params['curv'].value, dvdt_merr,
        dvdt_perr, model.time_base
    )

if __name__=="__main__":
    hack_radvel_plots = 0
    make_my_plot = 1
    main(hack_radvel_plots=hack_radvel_plots, make_my_plot=make_my_plot)
