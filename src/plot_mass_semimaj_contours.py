import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from cdips.plotting import savefig
import matplotlib as mpl
from astropy import units as u, constants as c
import matplotlib.patheffects as pe
from itertools import product

from scipy.ndimage import gaussian_filter
import logging
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

def format_ax(ax):
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')


def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, quiet=False,
           plot_datapoints=False, plot_density=False,
           plot_contours=True, no_fill_contours=False, fill_contours=True,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           pcolor_kwargs=None, **kwargs):
    """
    Plot a 2-D histogram of samples.
    (Function stolen from corner.py)

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    quiet : bool
        If true, suppress warnings for small datasets.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.

    """
    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2)
        # levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    # rgba_color = colorConverter.to_rgba(color)
    rgba_color = [0.0, 0.0, 0.0, 0.7]
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    if plot_contours or plot_density:
        # Compute the density levels.
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        if np.any(m) and not quiet:
            logging.warning("Too few points to create valid contours")
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()

        # Compute the bin centers.
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        # Extend the array for the sake of the contours at the plot edges.
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        #FIXME was H2.T
        ax.contourf(X2, Y2, H2, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        #FIXME: was H2.T
        ax.contourf(X2, Y2, H2, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        if pcolor_kwargs is None:
            pcolor_kwargs = dict()
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap, **pcolor_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        ax.contour(X2, Y2, H2, V, colors='k', linewidths=1)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])

    return ax


def plot_mass_semimaj_contours(prob_arr=None, mass_grid=None,
                                  sma_grid=None, with_contrast=False,
                                  discrete_color=False, linear_z=False,
                                  figpath=None, rvsonly=False):

    if prob_arr is None:

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

        rv_and_ao_log_like = np.log(nparr(np.exp(log_like_arr)*(1-ao_detected_arr)).mean(axis=2))

        # -2*logprob == chi^2
        # Convert likelihood values to a normalized probability via
        #   P ~ -exp(-chi^2/2)
        rv_prob_arr = np.exp(rv_log_like)/np.exp(rv_log_like).sum().sum()

        rv_and_ao_prob_arr = np.exp(rv_and_ao_log_like)/np.exp(rv_and_ao_log_like).sum().sum()

        prob_arr = rv_and_ao_prob_arr
        if rvsonly:
            prob_arr = rv_prob_arr

    # Sample from prob arr.

    # generate the set of all x,y pairs represented by the pmf
    pairs = nparr([(a.value,m.value) for a,m in product(
        sma_grid[:-1], mass_grid[:-1])])
    pairs = pairs.reshape((sma_grid.shape[0]-1, mass_grid.shape[0]-1, 2))

    # make random selections from the flattened pdf without replacement
    # whether you want replacement depends on your application
    N_samples = int(2e5)

    inds= np.random.choice(
        np.arange( (sma_grid.shape[0]-1) * (mass_grid.shape[0]-1)),
        p=prob_arr.reshape(-1), size=N_samples, replace=True
    )
    samples = pairs.reshape(-1,2)[inds]

    fig, ax = plt.subplots(figsize=(4,3))

    # samples = np.log10(samples)

    # smooth of 1.0 was ok
    bins = (sma_grid, mass_grid)
    ax = hist2d(
        samples[:,0], samples[:,1], bins=bins, range=None, weights=None,
        levels=None, smooth=2.0, ax=ax, color=None, quiet=False,
        plot_datapoints=False, plot_density=False, plot_contours=True,
        no_fill_contours=False, fill_contours=True, contour_kwargs=None,
        contourf_kwargs=None, data_kwargs=None, pcolor_kwargs=None
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    if with_contrast:

        from contrast_to_masslimit import get_companion_bounds
        zdf = get_companion_bounds('Zorro')

        dist_pc = 1/(3.7145e-3) # Bouma+2019, Table 1
        zdf['sma_AU'] = zdf.ang_sep * dist_pc

        ax.plot(
            nparr(zdf['sma_AU'])*u.AU,
            (nparr(zdf['m_comp/m_sun'])*u.Msun).to(u.Mjup),
            color='k', lw=1, ls=':',
            zorder=3, alpha=0.7
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

    ax.hlines(
        (0.864*u.Msun * 0.1**(1/3.5) ).to(u.Mjup).value, 3, 310, color='k',
        alpha=0.7, linestyles='--', linewidths=1
    )

    ax.set_xlabel('Semi-major axis [AU]')
    ax.set_ylabel('Companion mass [M$_\mathrm{{Jup}}$]')

    ax.set_ylim([1,1e3])
    ax.set_xlim([3,310])

    fig.tight_layout(h_pad=0, w_pad=0)

    format_ax(ax)

    dstr = '_discretecolor' if discrete_color else ''
    wstr = '_with_contrast' if with_contrast else ''
    lzstr = '_linearz' if linear_z else ''
    rvstr = '_rvonly' if rvsonly else ''
    if figpath is None:
        figpath = (
            '../results/mass_semimaj_contours{}{}{}{}.png'.
            format(wstr, dstr, lzstr, rvstr)
        )
    savefig(fig, figpath)


if __name__ == "__main__":
    for rvsonly in [True,False]:
        plot_mass_semimaj_contours(with_contrast=True, discrete_color=True,
                                   linear_z=False, rvsonly=rvsonly)
