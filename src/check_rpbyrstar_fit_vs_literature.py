# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy import units as u, constants as const

import os, argparse, pickle, h5py
from glob import glob
from numpy import array as nparr

def main(ticid, pickledir='/home/luke/Dropbox/proj/tessorbitaldecay/results/tess_lightcurve_fit_parameters/',
    sampledir='/home/luke/local/emcee_chains/',
    fittype='mandelagol_and_line',
    ndim=4):

    # from Southworth+ 2009 table 8:

    w08_rp = 1.416*u.Rjup
    w08_rstar = 0.937*u.Rsun

    g09_rp = 1.304*u.Rjup
    g09_rstar = 0.873*u.Rsun

    w09_rp = 1.365*u.Rjup
    w09_rstar = 0.912*u.Rsun

    s09_rp = 1.371*u.Rjup
    s09_rstar = 0.914*u.Rsun

    for paper, rp, rstar in zip(
        ['Wilson+08', 'Gillon+09', 'Winn+09', 'Southworth+09'],
        [w08_rp, g09_rp, w09_rp, s09_rp],
        [w08_rstar, g09_rstar, w09_rstar, s09_rstar]
    ):

        print('{:s}: Rp/Rstar = {:.4g}'.format(paper, (rp/rstar).cgs))

    # NOW GET UR MEASURED RP/RSTAR VALUES, COMPARE
    pickledir += str(ticid)
    fpattern = (
        '{:s}_{:s}_fit_empiricalerrs_t???.pickle'.
        format(str(ticid), fittype)
    )
    fnames = np.sort(glob(os.path.join(pickledir,fpattern)))
    samplepattern = (
        '{:s}_{:s}_fit_samples_{:d}d_t???_empiricalerrs.h5'.
        format(str(ticid), fittype, ndim)
    )
    samplenames = np.sort(glob(sampledir+samplepattern))

    rp_list, rp_bigerrs = (
        [],[]
    )

    transit_ix = 0
    for fname, samplename in zip(fnames, samplenames):
        transit_ix += 1

        d = pickle.load(open(fname, 'rb'))

        fitparams = d['fitinfo']['finalparams']
        fiterrs = d['fitinfo']['finalparamerrs']

        try:
            d = pickle.load(open(fname, 'rb'))

            fitparams = d['fitinfo']['finalparams']
            fiterrs = d['fitinfo']['finalparamerrs']

            rp_list.append(fitparams['rp'])
            rp_merrs = fiterrs['std_merrs']['rp']
            rp_perrs = fiterrs['std_perrs']['rp']
            rp_bigerrs.append( max((rp_merrs, rp_perrs)) )

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    rp, rp_bigerr = (
        nparr(rp_list),nparr(rp_bigerrs)
    )

    print('measured values from TESS:')
    for _rp, _rperr in zip(rp, rp_bigerr):
        print('{:.4f} +/- {:.4f}'.format(_rp, _rperr))

    print('average: {:.4f}'.format(np.mean(rp[rp_bigerr<0.01])))
    print('std: {:.4f}'.format(np.std(rp[rp_bigerr<0.01])))

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description=('Given a lightcurve with transits (e.g., alerted '
                     'from TESS Science Office), measure the times that they '
                     'fall at by fitting models.'))

    parser.add_argument('--ticid', type=int, default=None,
        help=('integer TIC ID for object. Pickle paramfile assumed to be at '
              '../results/tess_lightcurve_fit_parameters/'
             ))

    args = parser.parse_args()

    if not args.ticid:
        raise AssertionError('gotta give a ticid')

    main(args.ticid)
