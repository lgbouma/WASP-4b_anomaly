"""
For this study, we closely follow the procedure of \citet{bryan_excess_2019}.

We begin by defining a $50\times50$ grid in true planetary mass and semimajor
axis, evenly spaced from 1 to 300$\,$$M_{\rm Jup}$ and 3 to 500$\,$${\rm AU}$.

We then consider the possibility that an additional companion in any particular
cell can explain the observed linear trend.

In each cell, we consider 500 hypothetical companions.

We assign each companion a mass and semimajor axis from log-uniform
distributions within the grid cell. We draw the inclination from a uniform
distribution in $\cos i$.  The eccentricity is drawn from
\citet{kipping_beta_2013}'s long-period exoplanet Beta distribution ($a=1.12$,
$b=3.09$).

For each simulated companion, we then draw a sample from the converged chains
of our initial model of WASP-4b, plus its linear trend. We subtract
the planet's orbital solution, leaving the linear trend.

Given $(a_{\rm c}, M_{\rm c}, e_{\rm c}$ for each simulated outer companion,
and the fixed instrument offsets and jitters from the MCMC chains, we then
perform a maximum likelihood fit for the time and argument of periastron of the
outer simulated companion.

We then convert the resulting $50\times50\times500$ cube of best-fit
log-likelihood values to probabilities, and average over the samples in
each grid cell to derive a probability distribution in mass and semi-major
axis.

The distribution is shown in {\bf Figure~X}.
"""

###########
# imports #
###########

import os, pickle
import numpy as np, pandas as pd
from astropy import units as u, constants as c
from itertools import product
from numpy import array as nparr

import radvel
from radvel.mcmc import statevars
from radvel.driver import save_status, load_status
from radvel import driver

import configparser

from radvel_utils import (
    draw_models, _get_fit_results, args_object, initialize_sim_posterior
)
from scipy.stats import beta

from WASP4 import time_base

##########
# driver #
##########

def loguniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))

def derive_mass_semimaj_constraints(
    setupfn, rvfitdir, chainpath, verbose=True
    ):

    np.random.seed(42)

    chaindf = pd.read_csv(chainpath)

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, rvfitdir
    )

    n_grid_edges = 51 # a 4x4 grid has 5 edges. want: 51
    n_injections_per_cell = 500 # want: 500

    mass_grid = (
        np.logspace(np.log10(1), np.log10(300), num=n_grid_edges)*u.Mjup
    )
    sma_grid = (
        np.logspace(np.log10(3), np.log10(500), num=n_grid_edges)*u.AU
    )

    log_like_arr = np.zeros(
        (n_grid_edges-1, n_grid_edges-1, n_injections_per_cell)
    )

    sizestr = '{}x{}x{}'.format(n_grid_edges-1,
                                n_grid_edges-1,
                                n_injections_per_cell)
    outpath = (
        '../data/rv_simulations/mass_semimaj_loglikearr_{}.pickle'.
        format(sizestr)
    )

    if not os.path.exists(outpath):

        for mass_upper, sma_upper in product(mass_grid[1:], sma_grid[1:]):

            mass_left_ind = np.argwhere(mass_grid == mass_upper)-1
            sma_left_ind = np.argwhere(sma_grid == sma_upper)-1

            mass_lower = mass_grid[mass_left_ind].squeeze()
            sma_lower = sma_grid[sma_left_ind].squeeze()

            if verbose:
                print(mass_lower, mass_upper, sma_lower, sma_upper)

            # sample the models from the chain
            rv_model_single_planet_and_linear_trend, chain_sample_params = (
                draw_models(setupfn, rvfitdir, chaindf, rvtimes,
                            n_samples=n_injections_per_cell)
            )

            # n_sample x n_rvs array of the "RV - 1 planet" model (leaving in the
            # linear trend).
            # note: if there were a curvature term, it wuld go in here too.
            full_resid = (
                rvs - rv_model_single_planet_and_linear_trend
                + (
                    nparr(chain_sample_params.dvdt)[:,None]
                    *
                    (rvtimes[None,:] -time_base)
                )
            )

            # draw (a_c, M_c, e_c) for each simulated companion
            mass_c = loguniform(low=np.log10(mass_lower.value),
                                high=np.log10(mass_upper.value),
                                size=n_injections_per_cell)
            sma_c = loguniform(low=np.log10(sma_lower.value),
                               high=np.log10(sma_upper.value),
                               size=n_injections_per_cell)
            cos_incl_c = np.random.uniform(0, 1, size=n_injections_per_cell)
            incl_c = np.rad2deg(np.arccos(cos_incl_c))

            a, b = 1.12, 3.09
            ecc_c = beta.rvs(a, b, size=n_injections_per_cell)

            #
            # do a max-likelihood fit for time and argument of periastron.
            # 
            log_like_values = []
            for ix in range(n_injections_per_cell):

                data = pd.DataFrame({
                    'time': rvtimes,
                    'tel': telvec,
                    'mnvel': full_resid[ix, :],
                    'errvel': rverrs
                })

                gammajit_dict = {k:chain_sample_params.iloc[ix][k]
                                 for k in chain_sample_params.columns
                                 if 'gamma' in k or 'jit' in k}

                P, post = initialize_sim_posterior(
                    data, mass_c[ix]*u.Mjup, sma_c[ix]*u.AU,
                    incl_c[ix], ecc_c[ix], gammajit_dict
                )

                post = radvel.fitting.maxlike_fitting(post, verbose=False)

                if verbose:
                    print(post.logprob())

                log_like_values.append(post.logprob())

            log_like_arr[mass_left_ind, sma_left_ind, :] = (
                np.array(log_like_values)
            )

        with open(outpath, 'wb') as outf:
            pickle.dump(log_like_arr, outf, protocol=pickle.HIGHEST_PROTOCOL)

        print('saved {}'.format(outpath))

    else:

        log_like_arr = pickle.load(open(outpath, 'rb'))

    log_like = log_like_arr.mean(axis=2)

    prob_arr = np.exp(log_like)/np.exp(log_like).sum().sum()

    import IPython; IPython.embed()




if __name__=="__main__":

    # initialization script used to make the fix_gammadot fits
    basedir = os.path.join(
        os.path.expanduser('~'),
        "Dropbox/proj/WASP-4b_anomaly/"
    )

    setupfn = os.path.join(basedir,"src/WASP4.py")

    rvfitdir = os.path.join(
        basedir,
        "results/rv_fitting/LGB_20190911_fix_gammaddot"
    )

    chainpath = os.path.join(rvfitdir, 'WASP4_chains.csv.tar.bz2')

    derive_mass_semimaj_constraints(setupfn, rvfitdir, chainpath)
