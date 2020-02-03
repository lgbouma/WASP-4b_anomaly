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

from astrobase.lcmath import phase_magseries

from radvel_utils import args_object, _get_fit_results

def plot_rvs_2020_data_and_residual():

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
    savpath='../results/20190911_rv_data_and_residual.png'

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, outputdir
    )

    #
    # make the plot
    #
    offset=2450000

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

    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)

    assert curv==0
    model_line = dvdt*(_times-time_base)# + curv*(_times-time_base)**2

    model_merr = dvdt_merr*(_times-time_base)# + curv*(_times-time_base)**2
    model_perr = dvdt_perr*(_times-time_base)# + curv*(_times-time_base)**2

    a1.plot(_times-offset, model_line, color='black', zorder=-3, lw=0.5)
    a1.fill_between(_times-offset, model_merr, model_perr, color='black',
                    zorder=-4, alpha=0.2, lw=0)#label='$\pm 1\sigma$')
    a1.text(0.55, 0.54, 'Best-fit from RVs', va='bottom', ha='left',
            transform=a1.transAxes, color='black')

    # what would explain the Pdot from transits?
    period = 1.338231466*units.day
    Pdot_tra = -2.736e-10
    Pdot_err = 2**(1/2.)*2.83e-11 # inflating appropriately
    Pdot_tra_perr = Pdot_tra + Pdot_err
    Pdot_tra_merr = Pdot_tra - Pdot_err
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

    a1.plot(_mtimes-offset, model_tra_line-110,
            color='purple', zorder=-3, lw=0.5, ls=':')
    a1.fill_between(_mtimes-offset, model_tra_merr-110, model_tra_perr-110,
                    color='purple', zorder=-4, alpha=0.4, lw=0)
    a1.text(0.05, 0.12, 'Slope = $c\dot{P}/P$', va='bottom',
            ha='left', transform=a1.transAxes, color='purple', alpha=0.9)

    a0.legend(loc='upper right', fontsize='medium')

    a1.set_xlabel('Time [JD'+' - {}]'.format(offset), fontsize='large')
    a0.set_ylabel('Radial velocity [m/s]', fontsize='large')
    a1.set_ylabel('Residual [m/s]', fontsize='large')

    # make twin axis to show year on top
    times = Time(rvtimes, format='jd', scale='tdb')
    a_top = a0.twiny()
    a_top.scatter(times.decimalyear, rvs, s=0)
    a_top.set_xlabel('Year', fontsize='large')

    for a in [a0,a1]:
        a.set_xlim((3950, 9050))
    for a in [a0,a1,a_top]:
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')
    a0.tick_params(right=True, which='both', direction='in')
    a1.tick_params(top=True, right=True, which='both', direction='in')

    a0.set_ylim((-330, 330))
    a1.set_ylim((-230, 230))


    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)
    a_top.set_xlim(
        (Time( (3950+2450000), format='jd', scale='tdb').decimalyear,
        Time( (9050+2450000), format='jd', scale='tdb').decimalyear)
    )

    fig.tight_layout(h_pad=0.15, w_pad=0, pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


def plot_rvs_2020_orbit():

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
    savpath='../results/20190911_rv_data_and_residual.png'

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, outputdir
    )

    assert curv==0
    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)
    model_line = dvdt*(_times-time_base)# + curv*(_times-time_base)**2

    model_line_data = dvdt*(rvtimes-time_base)

    model_orbit = 0#FIXME gotta compute

    #NOTE: can't just subtract. the OFFSETS matter here too :-(
    #NOTE: this was actually an error in your earlier versions of this plot!
    data_orbit = rvs - 


    phase_magseries(times, rvs-, period, epoch, wrap=True, sort=True)

    #
    # make the plot
    #
    offset=2450000

    fig, ax = plt.subplots(nrows=1, ncols=1)

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

    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)

    assert curv==0
    model_line = dvdt*(_times-time_base)# + curv*(_times-time_base)**2

    model_merr = dvdt_merr*(_times-time_base)# + curv*(_times-time_base)**2
    model_perr = dvdt_perr*(_times-time_base)# + curv*(_times-time_base)**2

    a1.plot(_times-offset, model_line, color='black', zorder=-3, lw=0.5)
    a1.fill_between(_times-offset, model_merr, model_perr, color='black',
                    zorder=-4, alpha=0.2, lw=0)#label='$\pm 1\sigma$')
    a1.text(0.55, 0.54, 'Best-fit from RVs', va='bottom', ha='left',
            transform=a1.transAxes, color='black')

    # what would explain the Pdot from transits?
    period = 1.338231466*units.day
    Pdot_tra = -2.736e-10
    Pdot_err = 2**(1/2.)*2.83e-11 # inflating appropriately
    Pdot_tra_perr = Pdot_tra + Pdot_err
    Pdot_tra_merr = Pdot_tra - Pdot_err
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

    a1.plot(_mtimes-offset, model_tra_line-110,
            color='purple', zorder=-3, lw=0.5, ls=':')
    a1.fill_between(_mtimes-offset, model_tra_merr-110, model_tra_perr-110,
                    color='purple', zorder=-4, alpha=0.4, lw=0)
    a1.text(0.05, 0.12, 'Slope = $c\dot{P}/P$', va='bottom',
            ha='left', transform=a1.transAxes, color='purple', alpha=0.9)

    a0.legend(loc='upper right', fontsize='medium')

    a1.set_xlabel('Time [JD'+' - {}]'.format(offset), fontsize='large')
    a0.set_ylabel('Radial velocity [m/s]', fontsize='large')
    a1.set_ylabel('Residual [m/s]', fontsize='large')

    # make twin axis to show year on top
    times = Time(rvtimes, format='jd', scale='tdb')
    a_top = a0.twiny()
    a_top.scatter(times.decimalyear, rvs, s=0)
    a_top.set_xlabel('Year', fontsize='large')

    for a in [a0,a1]:
        a.set_xlim((3950, 9050))
    for a in [a0,a1,a_top]:
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')
    a0.tick_params(right=True, which='both', direction='in')
    a1.tick_params(top=True, right=True, which='both', direction='in')

    a0.set_ylim((-330, 330))
    a1.set_ylim((-230, 230))


    _times = np.linspace(np.min(rvtimes)-5000, np.max(rvtimes)+5000, num=2000)
    a_top.set_xlim(
        (Time( (3950+2450000), format='jd', scale='tdb').decimalyear,
        Time( (9050+2450000), format='jd', scale='tdb').decimalyear)
    )

    fig.tight_layout(h_pad=0.15, w_pad=0, pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))




if __name__=="__main__":

    plot_rvs_2020_data_and_residual()
    plot_rvs_2020_orbit()
