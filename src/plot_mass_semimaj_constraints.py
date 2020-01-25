import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from cdips.plotting import savefig
import matplotlib as mpl
from astropy import units as u, constants as c

def plot_mass_semimaj_constraints(prob_arr=None, mass_grid=None,
                                  sma_grid=None, with_contrast=False,
                                  discrete_color=False, linear_z=False):

    if prob_arr is None:

        n_mass_grid_edges = 76 # a 4x4 grid has 5 edges. want: 51
        n_sma_grid_edges = 51 # a 4x4 grid has 5 edges. want: 51
        n_injections_per_cell = 1 # 500 # want: 500

        mass_grid = (
            np.logspace(np.log10(1), np.log10(900), num=n_mass_grid_edges)*u.Mjup
        )
        sma_grid = (
            np.logspace(np.log10(3), np.log10(500), num=n_sma_grid_edges)*u.AU
        )

        sizestr = '{}x{}x{}'.format(n_mass_grid_edges-1,
                                    n_sma_grid_edges-1,
                                    n_injections_per_cell)
        pklpath = (
            '../data/rv_simulations/mass_semimaj_loglikearr_{}.pickle'.
            format(sizestr)
        )
        log_like_arr = pickle.load(open(pklpath, 'rb'))

        #
        # Convert log-likelihood values to relative probability by taking the exp.
        # Then average out the "sample" dimension (mass, sma, eccentricity, etc).
        #
        log_like = np.log(np.exp(log_like_arr).mean(axis=2))

        # -2*logprob == chi^2
        # Convert likelihood values to a normalized probability via
        #   P ~ -exp(-chi^2/2)
        prob_arr = np.exp(log_like)/np.exp(log_like).sum().sum()

    #################
    # make the plot #
    #################

    fig, ax = plt.subplots(figsize=(4,3))

    X,Y = np.meshgrid(sma_grid[:-1].value, mass_grid[:-1].value)

    cmap = plt.cm.gray_r

    if not linear_z:

        if discrete_color:
            bounds = np.round(np.linspace(-10, np.log(prob_arr).max(), 5),1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=np.log(prob_arr).max(), vmax=-10)

        im = ax.pcolormesh(X, Y, np.log(prob_arr), cmap=cmap, norm=norm,
                           shading='flat') # vs 'gouraud'

    else:

        if discrete_color:
            bounds = np.linspace(1e-10, prob_arr.max(), 5)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=prob_arr.max(), vmax=1e-10)

        im = ax.pcolormesh(X, Y, prob_arr, cmap=cmap, norm=norm)

    if with_contrast:

        from contrast_to_masslimit import get_companion_bounds
        zdf = get_companion_bounds()

        dist_pc = 1/(3.7145e-3) # Bouma+2019, Table 1
        zdf['sma_AU'] = zdf.ang_sep * dist_pc

        ax.plot(
            nparr(zdf['sma_AU'])*u.AU,
            (nparr(zdf['m_comp/m_sun'])*u.Msun).to(u.Mjup)
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Semi-major axis [AU]')
    ax.set_ylabel('Companion mass [M$_\mathrm{{jup}}$]')

    cbar = fig.colorbar(im, orientation='vertical', extend='min',
                        label='log(likelihood)')
    cbar.ax.tick_params(labelsize=8)

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    dstr = '_discretecolor' if discrete_color else ''
    wstr = '_with_contrast' if with_contrast else ''
    lzstr = '_linearz' if linear_z else ''
    figpath = (
        '../results/mass_semimaj_constraints{}{}{}.png'.
        format(wstr, dstr, lzstr)
    )
    savefig(fig, figpath)


if __name__ == "__main__":

    plot_mass_semimaj_constraints(with_contrast=True, discrete_color=True,
                                  linear_z=False)
    plot_mass_semimaj_constraints(with_contrast=False, discrete_color=True,
                                  linear_z=False)
