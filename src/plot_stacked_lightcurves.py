# -*- coding: utf-8 -*-
'''
make stacked lightcurve figure
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
import os, pickle

def plot_stacked_lightcurves(
    ticid, pickledir, excludetransits=None, xlim=None, ylim=None,
    offset=0.035):

    fnames = np.sort(glob(os.path.join(
        pickledir, '{:d}_*_empiricalerrs_*.pickle'.format(ticid))))

    n_transits = len(fnames)
    if isinstance(excludetransits,list):
        n_transits -= len(excludetransits)
    print('found {:d} transits'.format(n_transits))

    plt.close('all')
    f,(a0,a1) = plt.subplots(nrows=1, ncols=2, figsize=(4,11/2),
                             sharey=True)

    transit_ix, offset_ix = 0, 0
    for fname in fnames:

        if isinstance(excludetransits,list):
            if transit_ix in excludetransits:
                transit_ix += 1
                continue

        flux_offset = -offset*offset_ix

        try:
            d = pickle.load(open(fname, 'rb'))

            fit_epoch = d['fitinfo']['fitepoch'] # in BTJD

            time = (
                d['magseries']['times'] - fit_epoch
            )
            flux = d['magseries']['mags']
            err_flux = d['magseries']['errs']

            fit_time = time
            fit_flux = d['fitinfo']['fitmags']

            a0.scatter(time*24, flux+np.ones_like(flux)*flux_offset,
                       c='k', alpha=0.9, label='data', zorder=1, s=6,
                       rasterized=False, linewidths=0)
            a0.plot(
                fit_time*24, fit_flux+np.ones_like(flux)*flux_offset,
                c='b', zorder=0, rasterized=False, lw=1.5, alpha=0.4,
            )
            a1.scatter(time*24, 1+(flux-fit_flux)+np.ones_like(flux)*flux_offset,
                       c='k', alpha=0.9, label='data', zorder=1, s=6,
                       rasterized=False, linewidths=0)
            a1.plot(
                fit_time*24, 1+(fit_flux-fit_flux)+np.ones_like(flux)*flux_offset,
                c='b', zorder=0, rasterized=False, lw=1.5, alpha=0.4,
            )

            fiterrs = d['fitinfo']['finalparamerrs']
            t0_merrs = fiterrs['std_merrs']['t0']
            t0_perrs = fiterrs['std_perrs']['t0']
            t0_bigerrs = max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0'])

            if isinstance(xlim,list):
                #txt = '{:.6f}$\pm${:.6f}'.format(fit_epoch, t0_bigerrs)
                txt = '{:.2f}'.format(fit_epoch)
                a0.text(0.98*max(xlim),
                        np.median(fit_flux+np.ones_like(flux)*flux_offset)-2.5*np.std(fit_flux),
                        txt,
                        fontsize='xx-small',
                        va='center', ha='right')
                a1.text(0.98*max(xlim),
                        np.median(fit_flux+np.ones_like(flux)*flux_offset)-2.5*np.std(fit_flux),
                        txt,
                        fontsize='xx-small',
                        va='center', ha='right')

            transit_ix += 1
            offset_ix += 1

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    if isinstance(xlim,list):
        a0.set_xlim(xlim)
        a1.set_xlim(xlim)
    if isinstance(ylim,list):
        a0.set_ylim(ylim)
        a1.set_ylim(ylim)
    a0.set_ylabel('Relative flux')
    f.text(0.55,0, 'Time from mid-transit [hours]', ha='center')
    for ax in (a0,a1):
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    f.tight_layout(h_pad=0, w_pad=0.3)
    resultspng = '../results/{:d}_stacked_lightcurves.png'.format(ticid)
    paperpng = '../paper/f1.png'
    paperpdf = '../paper/f1.pdf'
    f.savefig(resultspng, dpi=400, bbox_inches='tight')
    print('saved {:s}'.format(resultspng))
    f.savefig(paperpng, dpi=400, bbox_inches='tight')
    print('saved {:s}'.format(paperpng))
    f.savefig(paperpdf, bbox_inches='tight')
    print('saved {:s}'.format(paperpdf))



def plot_stacked_lightcurves_single_col(
    ticid, pickledir, excludetransits=None, xlim=None, ylim=None,
    offset=0.035):

    fnames = np.sort(glob(os.path.join(
        pickledir, '{:d}_*_empiricalerrs_*.pickle'.format(ticid))))

    n_transits = len(fnames)
    if isinstance(excludetransits,list):
        n_transits -= len(excludetransits)
    print('found {:d} transits'.format(n_transits))

    plt.close('all')
    f,ax = plt.subplots(nrows=1, ncols=1,
                         figsize=(4,11/2))

    transit_ix, offset_ix = 0, 0
    for fname in fnames:

        if isinstance(excludetransits,list):
            if transit_ix in excludetransits:
                transit_ix += 1
                continue

        flux_offset = -offset*offset_ix

        try:
            d = pickle.load(open(fname, 'rb'))

            fit_epoch = d['fitinfo']['fitepoch'] # in BTJD

            time = (
                d['magseries']['times'] - fit_epoch
            )
            flux = d['magseries']['mags']
            err_flux = d['magseries']['errs']

            fit_time = time
            fit_flux = d['fitinfo']['fitmags']

            ax.scatter(time*24, flux+np.ones_like(flux)*flux_offset,
                       c='k', alpha=0.9, label='data', zorder=1, s=6,
                       rasterized=False, linewidths=0)

            ax.plot(
                fit_time*24, fit_flux+np.ones_like(flux)*flux_offset,
                c='b', zorder=0, rasterized=False, lw=1.5, alpha=0.4,
            )

            fiterrs = d['fitinfo']['finalparamerrs']
            t0_merrs = fiterrs['std_merrs']['t0']
            t0_perrs = fiterrs['std_perrs']['t0']
            t0_bigerrs = max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0'])

            if isinstance(xlim,list):
                #txt = '{:.6f}$\pm${:.6f}'.format(fit_epoch, t0_bigerrs)
                txt = '{:.2f}'.format(fit_epoch)
                ax.text(0.98*max(xlim),
                        np.median(fit_flux+np.ones_like(flux)*flux_offset)-2.5*np.std(fit_flux),
                        txt,
                        fontsize='xx-small',
                        va='center', ha='right')

            transit_ix += 1
            offset_ix += 1

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    if isinstance(xlim,list):
        ax.set_xlim(xlim)
    if isinstance(ylim,list):
        ax.set_ylim(ylim)
    ax.set_ylabel('Relative flux')
    ax.set_xlabel('Time from mid-transit [hour]')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    f.tight_layout(h_pad=0, w_pad=0)
    resultspng = '../results/{:d}_stacked_lightcurves.png'.format(ticid)
    paperpng = '../paper/f1.png'
    paperpdf = '../paper/f1.pdf'
    f.savefig(resultspng, dpi=400, bbox_inches='tight')
    print('saved {:s}'.format(resultspng))
    f.savefig(paperpng, dpi=400, bbox_inches='tight')
    print('saved {:s}'.format(paperpng))
    f.savefig(paperpdf, bbox_inches='tight')
    print('saved {:s}'.format(paperpdf))







if __name__=="__main__":

    #FIXME maybe better to argparse this...
    ticid = 402026209
    excludetransits = [9, 10]
    xlim = [-3, 3]
    ylim = [0.365, 1.01]
    offset = 0.035

    pickledir = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/results/'+
        'tess_lightcurve_fit_parameters/{:d}'.format(ticid)
    )

    plot_stacked_lightcurves(
        ticid, pickledir, excludetransits=excludetransits,
        offset=offset, xlim=xlim, ylim=ylim)
