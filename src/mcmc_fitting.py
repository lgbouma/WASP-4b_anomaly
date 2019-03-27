import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, argparse, pickle, h5py

from astrobase.timeutils import get_epochs_given_midtimes_and_period
from numpy import array as nparr
from parse import parse, search

from scipy import stats, optimize, integrate

from multiprocessing import Pool
import emcee
import corner

from astropy import units as u

if int(emcee.__version__[0]) >= 3:
    pass
else:
    raise AssertionError('require emcee > v3.0')


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


def log_prior(theta, plparams, delta_period=1e-5, delta_t0=3/(24*60)):
    """
      theta: can be length 2 or length 3 (linear, or quadratic models).

      theta[0]: t0, theta[1]: P, theta[2]: 1/2 dP/dE

      prob(t0) ~ U(t0-small number, t0+small number) [small # := 3 minutes]
      prob(P) ~ U(P-small number, P+small number) [small # := 1e-5 days]
      prob( 1/2 dP/dE ) ~ U( convert to Qstar! )

    from Eq 14 and 6 of Patra+ 2017,
    theta2 = 1/2 dP/dE = -1/2 P * 27*pi/(2*Qstar)*Mp/Mstar*(Rstar/a)^5.
    Qstar can be between say U[1e3,1e9].

    args:
        theta (np.ndarray): vector of parameters, in order listed above.
        plparams (tuple of floats): see order below.

    kwargs:
        delta_period (float): both in units of MINUTES, used to set bounds of
        uniform priors.
    """
    Rstar, Mplanet, Mstar, semimaj, nominal_period, nominal_t0 = plparams

    # now impose the prior on each parameter
    if len(theta)==2:
        t0, P = theta

        if ((nominal_period.value-delta_period < P <
             nominal_period.value+delta_period) and
            (nominal_t0.value-delta_t0 < t0 <
             nominal_t0.value+delta_t0)
           ):

            return 0.

        return -np.inf

    else:
        raise NotImplementedError


def log_likelihood(theta, data, data_occ=None):

    if len(theta)==2:
        model=linear_fit

    # unpack the data
    x, y, sigma_y = data
    if isinstance(data_occ,np.ndarray):
        x_occ, y_occ, sigma_y_occ = data_occ

    # evaluate the model at theta
    if not isinstance(data_occ,np.ndarray):
        y_fit = model(theta, x)
    else:
        y_fit, y_fit_occ = model(theta, x, x_occ=x_occ)

    # calculate the log likelihood
    if not isinstance(data_occ,np.ndarray):
        return -0.5 * np.sum(np.log(2 * np.pi * sigma_y ** 2)
                             + (y - y_fit) ** 2 / sigma_y ** 2)
    else:
        raise NotImplementedError


def log_posterior(theta, data, plparams, data_occ=None):

    theta = np.asarray(theta)

    lp = log_prior(theta, plparams)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, data, data_occ=data_occ)


def compute_mcmc(degree, data, plparams, theta_maxlike, plname, data_occ=None,
                 log_posterior=log_posterior, sampledir=None, n_walkers=50,
                 burninpercent=0.3, max_n_steps=10000,
                 overwriteexistingsamples=True, nworkers=8, plotcorner=True,
                 verbose=True, eps=1e-8, plotdir=None):

    if degree == 1:
        fitparamnames=['t0_days','P_days']
    else:
        raise NotImplementedError

    n_dim = degree + 1  # this determines the model

    samplesavpath = ( os.path.join(sampledir,
        plname+ '_degree{:d}_polynomial_timing_fit.h5'.format(degree)))

    backend = emcee.backends.HDFBackend(samplesavpath)
    if overwriteexistingsamples:
        print('erased samples previously at {:s}'.format(samplesavpath))
        backend.reset(n_walkers, n_dim)

    # if this is the first run, then start from a gaussian ball.
    # otherwise, resume from the previous samples.
    if isinstance(eps, float):
        starting_positions = ( theta_maxlike +
                               eps*np.random.randn(n_walkers, n_dim) )
    else:
        raise NotImplementedError

    isfirstrun = True
    if os.path.exists(backend.filename):
        if backend.iteration > 1:
            starting_positions = None
            isfirstrun = False

    if verbose and isfirstrun:
        print(
            'start MCMC with {:d} dims, max {:d} steps, {:d} walkers,'.format(
            n_dim, max_n_steps, n_walkers) + ' {:d} threads'.format(nworkers)
        )
    else:
        raise NotImplementedError

    # # works at ~50it/s
    # with Pool(nworkers) as pool:
    #     sampler = emcee.EnsembleSampler(
    #         n_walkers, n_dim, log_posterior,
    #         args=(data, plparams, data_occ),
    #         pool=pool,
    #         backend=backend
    #     )
    #     sampler.run_mcmc(starting_positions, max_n_steps,
    #                      progress=True)

    # ##########################################
    # # # works, 85 it/s
    # index = 0
    # autocorr = np.empty(max_n_steps)
    # old_tau = np.inf

    # sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
    #                                 args=(data, plparams, data_occ),
    #                                 backend=backend)

    # for sample in sampler.sample(starting_positions, iterations=max_n_steps,
    #                              progress=True):

    #     if sampler.iteration % 100:
    #         continue

    #     # compute autocorrleation time so far. tol=0 -> get an estimate,
    #     # even if it's not trustworthy
    #     tau = sampler.get_autocorr_time(tol=0)
    #     autocorr[index] = np.mean(tau)
    #     index += 1

    #     # check convergence
    #     converged = np.all(tau*100 < sampler.iteration)
    #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    #     if converged:
    #         break
    #     old_tau = tau
    #     print(tau)

    ##########################################
    # works, at ~50 it/s. so the pooling here slows things down! why?
    index = 0
    autocorr = np.empty(max_n_steps)
    old_tau = np.inf

    with Pool(nworkers) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_posterior,
            args=(data, plparams, data_occ),
            pool=pool,
            backend=backend
        )

        for sample in sampler.sample(starting_positions,
                                     iterations=max_n_steps, progress=True):

            if sampler.iteration % 100:
                continue

            # compute autocorrleation time so far. tol=0 -> get an estimate,
            # even if it's not trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # check convergence
            converged = np.all(tau*100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    ##########################################

    if verbose:
        print('ended MCMC run with max {:d} steps, {:d} walkers, '.
              format(max_n_steps, n_walkers)+
              '{:d} threads'.format(nworkers))

    reader = emcee.backends.HDFBackend(samplesavpath)
    n_steps_taken = reader.iteration

    n_to_discard = int(burninpercent*n_steps_taken)

    samples = reader.get_chain(discard=n_to_discard, flat=True)
    log_prob_samples = reader.get_log_prob(discard=n_to_discard, flat=True)
    log_prior_samples = reader.get_blobs(discard=n_to_discard, flat=True)

    # Get best-fit parameters, their 1-sigma error bars
    fit_statistics = list(
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            list(zip(*np.percentile(
                samples, [15.85, 50, 84], axis=0))))
    )

    medianparams, std_perrs, std_merrs = {}, {}, {}
    for ix, k in enumerate(fitparamnames):
        medianparams[k] = fit_statistics[ix][0]
        std_perrs[k] = fit_statistics[ix][1]
        std_merrs[k] = fit_statistics[ix][2]

    x, y, sigma_y = data
    if isinstance(data_occ,np.ndarray):
        x_occ, y_occ, sigma_y_occ = data_occ
    if not isinstance(data_occ,np.ndarray):
        returndict = {
            'fittype':'degree_{:d}_polynomial'.format(degree),
            'fitinfo':{
                'initial_guess':theta_maxlike,
                'maxlikeparams':theta_maxlike,
                'medianparams':medianparams,
                'std_perrs':std_perrs,
                'std_merrs':std_merrs
            },
            'samplesavpath':samplesavpath,
            'data':{
                'epoch':x,
                'tmid_minus_offset':y,
                'err_tmid':sigma_y,
            },
        }

    if plotcorner:
        if not plotdir:
            plotdir = os.path.join('../results')
        cornersavpath = os.path.join(
            plotdir, 'corner_degree_{:d}_polynomial.png'.format(degree))

        fig = corner.corner(
            samples,
            labels=fitparamnames,
            truths=theta_maxlike,
            quantiles=[0.16, 0.5, 0.84], show_titles=True
        )

        fig.savefig(cornersavpath, dpi=300)
        if verbose:
            print('saved {:s}'.format(cornersavpath))

    return returndict


def best_theta(degree, data, data_occ=None):
    """Standard frequentist approach: find the model that maximizes the
    likelihood under each model. Here, do it by direct optimization."""

    # create a zero vector of inital values
    theta_0 = np.zeros(degree+1)

    neg_logL = lambda theta: -log_likelihood(theta, data, data_occ=data_occ)

    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)


def linear_model(xdata, m, b):
    return m*xdata + b


def get_mcmc_timing_accuracy(plname, period_guess):

    # load in the data with ONLY the literature times. fit a linear ephemeris.
    manual_fpath = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/data/'+
        '{:s}_manual.csv'.format(plname)
    )
    mandf = pd.read_csv(manual_fpath, sep=';', comment=None)

    tmid = nparr(mandf['t0_BJD_TDB'])
    err_tmid = nparr(mandf['err_t0'])
    sel = np.isfinite(tmid) & np.isfinite(err_tmid)

    tmid = tmid[sel]
    err_tmid = err_tmid[sel]

    epoch, init_t0 = (
        get_epochs_given_midtimes_and_period(tmid, period_guess, verbose=True)
    )

    xdata = epoch
    ydata = tmid
    sigma = err_tmid

    data = nparr( [ epoch, tmid, err_tmid ] )

    # get max likelihood initial guess
    theta_linear = best_theta(1, data, data_occ=None)

    sampledir = '/home/luke/local/emcee_chains/wasp4b_line_fitting_check'
    plotdir = '../results/'
    if not os.path.exists(sampledir):
        os.mkdir(sampledir)
    plname = 'WASP-4b'
    plparams = 0, 0, 0, 0, 1.338231466*u.day, 2455073.841312848*u.day
    max_n_steps = 10000
    fit_2d = compute_mcmc(1, data, plparams, theta_linear, plname,
                          data_occ=None,
                          overwriteexistingsamples=True,
                          max_n_steps=max_n_steps,
                          sampledir=sampledir, nworkers=16,
                          plotdir=plotdir)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(
        linear_model, xdata, ydata, p0=(period_guess, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_period_err = pcov[0,0]**0.5
    lsfit_t0 = popt[1]
    lsfit_t0_err = pcov[1,1]**0.5

    print('\n')
    print('LEAST SQUARES GIVES')
    print('period: {:.10f},   t0: {:.10f}'.format(lsfit_period, lsfit_t0))
    print('period_err: {:.3e},   t0_err: {:.3e}'.
          format(lsfit_period_err, lsfit_t0_err))

    print('\n')
    print('MCMC GIVES')
    print(fit_2d['fitinfo']['medianparams'])
    print(fit_2d['fitinfo']['std_perrs'])
    print(fit_2d['fitinfo']['std_merrs'])

    print('\n')

    return 1
