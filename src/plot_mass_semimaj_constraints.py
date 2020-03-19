import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from cdips.plotting import savefig
import matplotlib as mpl
from astropy import units as u, constants as c
import matplotlib.patheffects as pe

def plot_mass_semimaj_constraints(prob_arr=None, mass_grid=None,
                                  sma_grid=None, with_contrast=False,
                                  discrete_color=False, linear_z=False,
                                  figpath=None):

    if prob_arr is None:

        # n_mass_grid_edges = 31 # a 4x4 grid has 5 edges. want: 64+1, 128+1...
        # n_sma_grid_edges = 31 # a 4x4 grid has 5 edges.
        # n_injections_per_cell = 16 # 500 # want: 500
        n_mass_grid_edges = 129 # a 4x4 grid has 5 edges. want: 51
        n_sma_grid_edges = 129 # a 4x4 grid has 5 edges. want: 51
        n_injections_per_cell = 512 # 500 # want: 500

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
        ao_detected_arr = pickle.load(
            open(pklpath.replace('loglikearr', 'aodetectedarr'), 'rb')
        )

        #
        # Convert log-likelihood values to relative probability by taking the exp.
        # Then average out the "sample" dimension (mass, sma, eccentricity, etc).
        #
        rv_log_like = np.log(np.exp(log_like_arr).mean(axis=2))

        rv_and_ao_log_like = np.log(np.exp(log_like_arr*(1-ao_detected_arr)).mean(axis=2))

        # -2*logprob == chi^2
        # Convert likelihood values to a normalized probability via
        #   P ~ -exp(-chi^2/2)
        rv_prob_arr = np.exp(rv_log_like)/np.exp(rv_log_like).sum().sum()

        rv_and_ao_prob_arr = np.exp(rv_and_ao_log_like)/np.exp(rv_and_ao_log_like).sum().sum()

        prob_arr = rv_and_ao_prob_arr

    #################
    # make the plot #
    #################

    fig, ax = plt.subplots(figsize=(4,3))

    X,Y = np.meshgrid(sma_grid[:-1].value, mass_grid[:-1].value)

    cmap = plt.cm.gray_r

    if not linear_z:

        cutoff = -6.3

        if discrete_color:
            bounds = np.round(np.linspace(cutoff, np.log10(prob_arr).max(), 5),1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=np.log10(prob_arr).max(),
                                        vmax=cutoff)

        im = ax.pcolormesh(X, Y, np.log10(prob_arr), cmap=cmap, norm=norm,
                           shading='flat') # vs 'gouraud'

    else:

        if discrete_color:
            bounds = np.linspace(10**(cutoff), prob_arr.max(), 5)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=prob_arr.max(),
                                        vmax=10**(cutoff))

        im = ax.pcolormesh(X, Y, prob_arr, cmap=cmap, norm=norm)

    if with_contrast:

        from contrast_to_masslimit import get_companion_bounds
        zdf = get_companion_bounds('Zorro')

        dist_pc = 1/(3.7145e-3) # Bouma+2019, Table 1
        zdf['sma_AU'] = zdf.ang_sep * dist_pc

        ax.plot(
            nparr(zdf['sma_AU'])*u.AU,
            (nparr(zdf['m_comp/m_sun'])*u.Msun).to(u.Mjup),
            color='C0', lw=1,
            zorder=4
        )

        # NOTE: not really ruled out!
        # t = ax.text(
        #     100, 450, 'Ruled out by\nspeckle imaging',
        #     fontsize=7.5, ha='center', va='center',
        #     # path_effects=[pe.withStroke(linewidth=0.5, foreground="white")],
        #     color='C0', zorder=3
        # )

        # ax.fill_between(
        #     nparr(zdf['sma_AU'])*u.AU,
        #     (nparr(zdf['m_comp/m_sun'])*u.Msun).to(u.Mjup),
        #     1000,
        #     color='white',
        #     alpha=0.8,
        #     zorder=2
        # )

        _sma = np.logspace(np.log10(90), np.log10(180))
        _mass_min = 2 # rescale
        k = _mass_min / min(_sma**2)
        _mass = k*_sma**2
        ax.plot(
            _sma, _mass, color='k', lw=1
        )
        ax.text(
            np.percentile(_sma, 60), np.percentile(_mass, 40), '$M\propto a^2$',
            fontsize=7.5, ha='left', va='center',
            path_effects=[pe.withStroke(linewidth=0.5, foreground="white")],
            color='black'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Semi-major axis [AU]')
    ax.set_ylabel('Companion mass [M$_\mathrm{{Jup}}$]')

    cbar = fig.colorbar(im, orientation='vertical', extend='min',
                        label='$\log_{{10}}$(probability)')
    cbar.ax.tick_params(labelsize=6, direction='out')

    ax.set_ylim([1,1e3])
    ax.set_xlim([3,310])

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.tick_params(right=True, which='both', direction='in')
    ax.tick_params(top=True, which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    dstr = '_discretecolor' if discrete_color else ''
    wstr = '_with_contrast' if with_contrast else ''
    lzstr = '_linearz' if linear_z else ''
    if figpath is None:
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
