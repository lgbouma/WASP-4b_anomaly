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

import os, pickle, configparser
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import units as u, constants as const

from itertools import product
from numpy import array as nparr
from datetime import datetime
import multiprocessing as mp

import radvel
from radvel.mcmc import statevars
from radvel.driver import save_status, load_status
from radvel.kepler import kepler
from radvel import driver

from scipy.stats import beta, powerlaw
from scipy.interpolate import interp1d

from cdips.plotting import savefig
# from WASP4 import time_base
from radvel_utils import (
    draw_models, _get_fit_results, args_object, initialize_sim_posterior
)
from plot_mass_semimaj_constraints import plot_mass_semimaj_constraints
from contrast_to_masslimit import get_companion_bounds

##########
# config #
##########

VERBOSE = 0

INCLUDE_AO = 1  # whether to include the AO observations in the likelihood
t_obs_AO = 2458745.5 # Sep 28, 2019 -> JD.

# WASP-4 config parameters
MSTAR = 0.864*u.Msun
DSTAR = 1/(3.7145e-3)*u.pc

# columns: ang_sep (in arcsec), delta_mag, m_comp/m_sun
zdf = get_companion_bounds('Zorro')
interp_fn = interp1d(nparr(zdf['ang_sep']), nparr(zdf['m_comp/m_sun']),
                     bounds_error=False, fill_value='extrapolate')

####################
# helper functions #
####################

def loguniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))


def _ξ(a, mplanet, E, e, ω, i, Ω=0):

    assert isinstance(mplanet, u.quantity.Quantity)
    assert isinstance(a, u.quantity.Quantity)

    # maximal apparent angular extent of orbit. units: arcsec (note a_AU/d_pc
    # gives an angle in arcseconds).  e.g., 2020/03/19.1 derivation, or Eq 61
    # of Murray & Correia 2010.
    θ = ( (MSTAR / (mplanet+MSTAR)).cgs.value *
          (a.to(u.AU)/DSTAR.to(u.pc)).value
    )*u.arcsec

    return _B(θ,ω,Ω,i) * _X(E,e) + _G(θ,ω,Ω,i) * _Y(E,e)


def _η(a, mplanet, E, e, ω, i, Ω=0):

    assert isinstance(mplanet, u.quantity.Quantity)
    assert isinstance(a, u.quantity.Quantity)

    θ = ( (MSTAR / (mplanet+MSTAR)).cgs.value *
          (a.to(u.AU)/DSTAR.to(u.pc)).value
    )*u.arcsec


    return _A(θ,ω,Ω,i) * _X(E,e) + _F(θ,ω,Ω,i) * _Y(E,e)


def _A(θ, ω, Ω, i):
    # Quirrenbach 2010 Eqns 4-7
    return (
        θ*(
            np.cos(ω)*np.cos(Ω) - np.sin(ω)*np.sin(Ω)*np.cos(i)
        )
    )

def _B(θ, ω, Ω, i):
    # Quirrenbach 2010 Eqns 4-7
    return (
        θ*(
            np.cos(ω)*np.sin(Ω) + np.sin(ω)*np.cos(Ω)*np.cos(i)
        )
    )

def _F(θ, ω, Ω, i):
    # Quirrenbach 2010 Eqns 4-7
    return (
        θ*(
            -np.sin(ω)*np.cos(Ω) - np.cos(ω)*np.sin(Ω)*np.cos(i)
        )
    )

def _G(θ, ω, Ω, i):
    # Quirrenbach 2010 Eqns 4-7
    return (
        θ*(
            -np.sin(ω)*np.sin(Ω) + np.cos(ω)*np.cos(Ω)*np.cos(i)
        )
    )


def _X(E, e):
    return np.cos(E) - e


def _Y(E, e):
    return np.sqrt(1-e*e) * np.sin(E)


###########
# drivers #
###########

def maxlike_worker(task):

    data, gammajit_dict, mass, sma, incl, ecc = task

    # initialize posterior and solve for time and argument of periapse.
    post = initialize_sim_posterior(
        data, mass*u.Mjup, sma*u.AU, incl, ecc, gammajit_dict
    )

    post = radvel.fitting.maxlike_fitting(post, verbose=VERBOSE)

    is_AO_detectable = get_AO_detectability(mass, sma, incl, ecc, post)

    return (post.logprob(), is_AO_detectable)


def get_AO_detectability(mass, sma, incl, ecc, post):

    # orbital period
    period = ((
        (sma*u.AU)**(3) *
        4*np.pi**2 / (const.G * (MSTAR + mass*u.Mjup) )
    )**(1/2)).to(u.day).value

    # observed time
    t = t_obs_AO

    w1, tc1 = post.get_vary_params()

    # reference time at periapse (units: days)
    tp = tc1

    # argument of periapse. (units: radians)
    ωp = w1

    # mean anomaly, e.g., as calculated in radvel.orbit.true_anomaly, in units
    # of radians.
    M = 2 * np.pi * (((t - tp) / period) - np.floor((t - tp) / period))

    # eccentric anomaly via Kepler eqn, in radians.
    E = kepler(nparr([float(M)]), nparr([float(ecc)]))

    # on-sky separation in direction aligned with line of nodes.
    # equations defined in Quirrenbach (2010) exoplanets review.
    ξ = _ξ(sma*u.AU, mass*u.Mjup, E, ecc, ωp, np.deg2rad(incl), Ω=0)

    # on-sky separation in direction perpendicular to line of nodes.
    η = _η(sma*u.AU, mass*u.Mjup, E, ecc, ωp, np.deg2rad(incl), Ω=0)

    # projected separation, in arcsec
    proj_sep = (np.sqrt( ξ**2 + η**2 )).to(u.arcsec).value

    # convert projected separation and mass to detection 
    max_detectable_mass = interp_fn(proj_sep)*u.Msun
    is_AO_detectable = int(mass*u.Mjup > max_detectable_mass)

    return is_AO_detectable


def derive_mass_semimaj_constraints(
    setupfn, rvfitdir, chainpath, verbose=False,
    multiprocess=True
    ):

    np.random.seed(42)

    chaindf = pd.read_csv(chainpath)

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, rvfitdir
    )

    # #NOTE: debugging
    # n_mass_grid_edges = 31 # a 4x4 grid has 5 edges. want: 64+1, 128+1...
    # n_sma_grid_edges = 31 # a 4x4 grid has 5 edges.
    # n_injections_per_cell = 16 # 500 # want: 500
    # NOTE: production
    n_mass_grid_edges = 129 # a 4x4 grid has 5 edges. want: 64+1, 128+1...
    n_sma_grid_edges = 129 # a 4x4 grid has 5 edges.
    n_injections_per_cell = 512 # 500 # want: 500

    mass_grid = (
        np.logspace(np.log10(1), np.log10(900), num=n_mass_grid_edges)*u.Mjup
    )
    sma_grid = (
        np.logspace(np.log10(3), np.log10(500), num=n_sma_grid_edges)*u.AU
    )

    log_like_arr = np.zeros(
        (n_mass_grid_edges-1, n_sma_grid_edges-1, n_injections_per_cell)
    )
    ao_detected_arr = np.zeros_like(log_like_arr)

    sizestr = '{}x{}x{}'.format(n_mass_grid_edges-1,
                                n_sma_grid_edges-1,
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

            pstr = (
                '{:s} {:.2f}, {:.2f}, {:.2f}, {:.2f}'.
                format(datetime.now().isoformat(),
                       mass_lower, mass_upper, sma_lower, sma_upper)
            )
            print(pstr)

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
                    (rvtimes[None,:] - time_base)
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

            ecc_c = np.zeros(n_injections_per_cell)

            # case: companion is a planet. use Kipping (2013).
            a, b = 1.12, 3.09
            _ecc_c_planetary = beta.rvs(a, b, size=n_injections_per_cell)

            # case: companion is a star. go for Moe & Di Stefano (2017), Eq 17.

            period_c = ((
                (sma_c*u.AU)**(3) *
                4*np.pi**2 / (const.G * (MSTAR+mass_c*u.Mjup) )
            )**(1/2)).to(u.day)

            # Moe & Di Stefano (2017), Eq 17.
            eta = 0.6 - 0.7 / ( np.log10(period_c.to(u.day).value) - 0.5 )

            # f(x,a) = ax^{a-1} for scipy's built-in powerlaw distribution.
            _ecc_c_stellar = powerlaw.rvs(eta+1, size=n_injections_per_cell)

            # assign the eccentricties piece-wise at 10 Mjup.
            cutoff = 10 # Mjup
            ecc_c[mass_c <= cutoff] = _ecc_c_planetary[mass_c <= cutoff]
            ecc_c[mass_c > cutoff] = _ecc_c_stellar[mass_c > cutoff]

            #
            # do a max-likelihood fit for time and argument of periastron.
            # 

            if multiprocess:
                tasks = [(
                    pd.DataFrame({'time': rvtimes, 'tel': telvec, 'mnvel':
                                  full_resid[ix, :], 'errvel': rverrs }),
                    {k: chain_sample_params.iloc[ix][k] for k in
                     chain_sample_params.columns if 'gamma' in k or 'jit' in k},
                    mass_c[ix],
                    sma_c[ix],
                    incl_c[ix],
                    ecc_c[ix]) for ix in range(n_injections_per_cell)
                ]

                nworkers = mp.cpu_count()
                maxworkertasks = 1000
                pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

                result = pool.map(maxlike_worker, tasks)

                pool.close()
                pool.join()

                loglike_vals, ao_detected_vals = (
                    nparr(result)[:,0], nparr(result)[:,1]
                )

            if not multiprocess:
                # serial approach
                loglike_vals, ao_detected_vals = [], []
                for ix in range(n_injections_per_cell):

                    data = pd.DataFrame({
                        'time': rvtimes,
                        'tel': telvec,
                        'mnvel': full_resid[ix, :],
                        'errvel': rverrs
                    })

                    gammajit_dict = {k: chain_sample_params.iloc[ix][k]
                                     for k in chain_sample_params.columns
                                     if 'gamma' in k or 'jit' in k}

                    post = initialize_sim_posterior(
                        data, mass_c[ix]*u.Mjup, sma_c[ix]*u.AU,
                        incl_c[ix], ecc_c[ix], gammajit_dict
                    )

                    post = radvel.fitting.maxlike_fitting(post, verbose=False)

                    ao_detected = get_AO_detectability(
                        mass_c[ix], sma_c[ix], incl_c[ix], ecc_c[ix], post
                    )

                    if verbose:
                        print(post.logprob())

                    loglike_vals.append(post.logprob())
                    ao_detected_vals.append(ao_detected)

            log_like_arr[mass_left_ind, sma_left_ind, :] = (
                nparr(loglike_vals)
            )

            ao_detected_arr[mass_left_ind, sma_left_ind, :] = (
                nparr(ao_detected_vals)
            )

            ostr = (
                '\t loglike: {:.1f}. ao_detected_frac: {:.2f}'.
                format(np.nanmean(nparr(loglike_vals)),
                       np.nanmean(nparr(ao_detected_vals)))
            )
            print(ostr)


        with open(outpath, 'wb') as outf:
            pickle.dump(log_like_arr, outf, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved {}'.format(outpath))

        outpath = outpath.replace('loglikearr', 'aodetectedarr')
        with open(outpath, 'wb') as outf:
            pickle.dump(ao_detected_arr, outf, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved {}'.format(outpath))

    else:

        log_like_arr = pickle.load(
            open(outpath, 'rb')
        )
        ao_detected_arr = pickle.load(
            open(outpath.replace('loglikearr', 'aodetectedarr'), 'rb')
        )

    #
    # Convert log-likelihood values to relative probability by taking the exp.
    # Then average out the "sample" dimension (mass, sma, eccentricity, etc).
    #
    rv_log_like = np.log(np.exp(log_like_arr).mean(axis=2))

    rv_and_ao_log_like = np.log(nparr(np.exp(log_like_arr)*(1-ao_detected_arr)).mean(axis=2))

    # -2*logprob == chi^2
    # Convert likelihood values to a normalized probability via
    #   P ~ -exp(-chi^2/2)
    rv_prob_arr = np.exp(rv_log_like)/np.exp(rv_log_like).sum().sum()

    rv_and_ao_prob_arr = np.exp(rv_and_ao_log_like)/np.exp(rv_and_ao_log_like).sum().sum()

    return rv_prob_arr, rv_and_ao_prob_arr, mass_grid, sma_grid


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

    rv_prob_arr, rv_and_ao_prob_arr, mass_grid, sma_grid = (
        derive_mass_semimaj_constraints(setupfn, rvfitdir, chainpath,
                                        verbose=VERBOSE)
    )

    figpath = '../results/mass_semimaj_constraints_rvonly.png'
    plot_mass_semimaj_constraints(rv_prob_arr, mass_grid, sma_grid,
                                  figpath=figpath)

    figpath = '../results/mass_semimaj_constraints_rvandao.png'
    plot_mass_semimaj_constraints(rv_and_ao_prob_arr, mass_grid, sma_grid,
                                  figpath=figpath)
