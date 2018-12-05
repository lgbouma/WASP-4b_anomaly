# -*- coding: utf-8 -*-
'''
three subplots:
  * eccentricity damping timescale (like Bonomo+2017)
  * orbital decay timescale
  * HR diagram.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from numpy import array as nparr

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const
from astropy import units

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

import itertools
import os
from shutil import copyfile

def get_bonomo_2017_mergetable():

    t7 = Table.read("../data/Bonomo_2017_table7.vot", format="votable")
    t8 = Table.read("../data/Bonomo_2017_table8.vot", format="votable")
    t9 = Table.read("../data/Bonomo_2017_table9.vot", format="votable")

    df7 = t7.to_pandas()
    df8 = t8.to_pandas()
    df9 = t9.to_pandas()

    # fix the unicode
    for k in ['Planet','Forbit','Fcirc','Fcomp']:
        df8[k] = list(map(lambda x: x.decode(), df8[k]))
    for k in ['Star','Ref','n_Ref']:
        df7[k] = list(map(lambda x: x.decode(), df7[k]))

    planets = nparr([''.join(p.split()) for p in nparr(df8['Planet'])])
    stars = nparr(df7['Star'])

    Mstar = nparr(df7['Mstar'])
    Rstar = nparr(df7['Rstar'])
    Teff = nparr(df7['Teff'])

    Mplanet = nparr(df9['Mp'])
    Rplanet = nparr(df7['Rplanet'])

    sma = nparr(df9['smaxis'])
    period = nparr(df9['Period'])

    forbit = nparr(df8['Forbit'])
    fcirc = nparr(df8['Fcirc'])
    fcomp = nparr(df8['Fcomp'])

    df7_planets = nparr([''.join(s.split()) + 'b' for s in stars])

    try:
        np.testing.assert_array_equal(planets, df7_planets)
    except AssertionError:
        print(np.setdiff1d(df7_planets, planets))
        print(np.setdiff1d(planets, df7_planets))

    df = pd.DataFrame({
        'planet':planets,
        'star':stars,
        'Mstar':Mstar,
        'Rstar':Rstar,
        'Teff':Teff,
        'Mplanet':Mplanet,
        'Rplanet':Rplanet,
        'sma':sma,
        'period':period,
        'forbit':forbit,
        'fcirc':fcirc,
        'fcomp':fcomp,
        'ecc':nparr(df8['ecc']),
        'ecc_merr':nparr(df8['e_ecc']),
        'ecc_perr':nparr(df8['E_ecc'])
    })

    return df

def plot_ecc_damping_timescale(df, ax, Qplanetprime=1e5):
    """
    from 2018/12/04.3, equation (1): (which all started from Metzger+2012)

    tau_e = ( 2*Qplanetprime / (63*np.pi) *
              (sma/Rplanet)**5 *
              (Mplanet/Mstar) *
              P
    ).to(u.Myr)


    thus

    (Mplanet/Mstar) * P = 63*np.pi/(2*Qplanetprime) * tau_e * (Rplanet/sma)**5
    """

    sma = nparr(df['sma'])*u.au
    Rplanet = nparr(df['Rplanet'])*u.Rjup
    Mplanet = nparr(df['Mplanet'])*u.Mjup
    Mstar = nparr(df['Mstar'])*u.Msun
    period = nparr(df['period'])*u.day

    forbit = nparr(df['forbit'])
    is_ecc = (forbit == 'E')

    planets = nparr(df['planet'])
    # my planet list:
    plnames=['WASP-4b','WASP-5b','WASP-6b','WASP-12b','WASP-18b','WASP-46b']
    if not len(np.setdiff1d(plnames, planets))==0:
        print('error: need all planets to be in B+17 table')
        print(np.setdiff1d(plnames, planets))
        raise AssertionError

    y = (period * Mplanet/Mstar).to(u.day).value
    x = (sma/Rplanet).cgs.value

    x_grid = np.logspace(1,3,num=1000)
    tau_e = 10*u.Myr
    y_10myr = (63*np.pi/(2*Qplanetprime)*tau_e*(1/x_grid)**5).to(u.day).value

    tau_e = 100*u.Myr
    y_100myr = (63*np.pi/(2*Qplanetprime)*tau_e*(1/x_grid)**5).to(u.day).value

    ax.scatter(x[~is_ecc], y[~is_ecc], marker='o', s=5, zorder=2,
               facecolors='none', edgecolors='k', lw=0.4,
               label='no eccentricity detected', alpha=0.3)
    ax.scatter(x[is_ecc], y[is_ecc], marker=',', s=5, zorder=2,
               color='k',
               label='significantly eccentric (RV)')

    marker = itertools.cycle(('o', 'v', '>', 'D', 's', 'P'))
    for tesspl in plnames:
        if tesspl=='WASP-4b':
            s=56
        else:
            s=28
        ax.scatter(x[planets==tesspl], y[planets==tesspl],
                   marker=next(marker), s=s, zorder=3,
                   edgecolor='black', lw=0.4)

    ax.plot(x_grid, y_10myr, zorder=-2, lw=1, c='#1f77b4', ls='-')
    ax.text(92, 3e-4, '$\\tau_e = 10\ \mathrm{Myr}$', va='center',ha='center',
            fontsize='xx-small', rotation=295)
    ax.plot(x_grid, y_100myr, zorder=-2, lw=1, c='#1f77b4', ls='--')
    ax.text(184, 3e-4, '$\\tau_e = 100\ \mathrm{Myr}$', va='center',ha='center',
            fontsize='xx-small', rotation=295)

    leg = ax.legend(loc='upper left', fontsize='x-small', framealpha=1)

    ax.set_xlabel('$a/R_{\mathrm{p}}$')
    ax.set_ylabel(r'$P \times M_{\mathrm{p}} / M_{\star}$ [days]')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    ax.set_xlim((9.5,1050))
    ax.set_ylim((1e-4,1))


def plot_orbital_decay_timescale(df, ax, Qstarprime=1e6):
    """
    from 2018/12/04.3, equation (2): (which all started from Metzger+2012)

    tau_a = P * (Qstarprime/(9*pi)) * (Mstar/Mp) * (a/Rstar)^5

    so you plot:

    P * Mstar/Mp = tau_a * (9*np.pi)/Qstarprime * (Rstar/a)**5

    or set
    x = (a/Rstar)^5,
    y = (Mstar/Mplanet)*period
    """

    sma = nparr(df['sma'])*u.au
    Rplanet = nparr(df['Rplanet'])*u.Rjup
    Mplanet = nparr(df['Mplanet'])*u.Mjup
    Mstar = nparr(df['Mstar'])*u.Msun
    Rstar = nparr(df['Rstar'])*u.Rsun
    period = nparr(df['period'])*u.day

    forbit = nparr(df['forbit'])
    is_ecc = (forbit == 'E')

    planets = nparr(df['planet'])
    # my planet list:
    plnames=['WASP-4b','WASP-5b','WASP-6b','WASP-12b','WASP-18b','WASP-46b']
    if not len(np.setdiff1d(plnames, planets))==0:
        print('error: need all planets to be in B+17 table')
        print(np.setdiff1d(plnames, planets))
        raise AssertionError

    y = ((Mstar/Mplanet)*period).to(u.day).value
    x = ((sma/Rstar)).cgs.value

    x_grid = np.logspace(0,2,num=1000)
    tau_a = 100*u.Myr
    y_100myr = ( tau_a * (9*np.pi)/Qstarprime * (1/x_grid)**5 ).to(u.day)

    tau_a = 1*u.Gyr
    y_1gyr = ( tau_a * (9*np.pi)/Qstarprime * (1/x_grid)**5 ).to(u.day)

    ax.scatter(x[~is_ecc], y[~is_ecc], marker='o', s=5, zorder=2,
               facecolors='none', edgecolors='k', lw=0.4, alpha=0.3)
    ax.scatter(x[is_ecc], y[is_ecc], marker=',', s=5, zorder=2,
               color='k')

    marker = itertools.cycle(('o', 'v', '>', 'D', 's', 'P'))
    for tesspl in plnames:
        if tesspl=='WASP-4b':
            s=56
        else:
            s=28
        ax.scatter(x[planets==tesspl], y[planets==tesspl],
                   marker=next(marker), s=s, zorder=3,
                   edgecolor='black', lw=0.4)

    ax.plot(x_grid, y_100myr, zorder=-2, lw=1, c='#1f77b4', ls='-')
    ax.text(1.6, 5e4, '$\\tau_a = 100\ \mathrm{Myr}$', va='center',ha='center',
            fontsize='xx-small', rotation=295)
    ax.plot(x_grid, y_1gyr, zorder=-2, lw=1, c='#1f77b4', ls='--')
    ax.text(3.3, 5e4, '$\\tau_a = 1\ \mathrm{Gyr}$', va='center',ha='center',
            fontsize='xx-small', rotation=295)

    ax.set_xlabel('$a/R_\star$')
    ax.set_ylabel(r'$P \times M_{\star} / M_{\mathrm{p}}$ [days]')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    ax.set_xlim((1e0,1.2e2))
    ax.set_ylim((3.3e1,2.3e5))


def do_gaia_match(df):

    # first, crossmatch against SIMBAD to get the coordinates

    starnames = nparr(df['star'])
    for ix, s in enumerate(starnames):
        # manual string subbing for simbad query
        if 'Qatar' in s:
            starnames[ix] = s.replace('-',' ')
        if s == 'WASP-70A':
            starnames[ix] = s.rstrip('A')

    ras, decs = [], []
    print('running SIMBAD query...')
    for ix, starname in enumerate(starnames):
        print(ix, starname)
        result = Simbad.query_object(starname)
        if len(result) != 1:
            raise AssertionError
        ras.append(result['RA'].tolist()[0])
        decs.append(result['DEC'].tolist()[0])
    ras = np.array(ras)
    decs = np.array(decs)

    coords = SkyCoord(ras, decs, frame='icrs', unit=(u.hourangle, u.deg))

    # then, crossmatch against Gaia to get the absolute G mags

    radius = units.Quantity(2, units.arcsec)

    sep, gaia_id, parallax, gmag, dr2_teff, dr2_radius, dr2_lum, distance = (
        [],[],[],[],[],
        [],[],[]
    )
    print('running Gaia query...')
    for ix, sysname, coord in zip(range(len(coords)), nparr(df['star']), coords):
        print('{:d}/{:d} --- {:s}'.format(ix, len(coords), sysname))
        j = Gaia.cone_search_async(coord, radius)
        r = j.get_results()

        if len(r) == 0:
            print('\tno match, skipping')
            for l in [sep, gaia_id, parallax, gmag, dr2_teff, dr2_radius,
                      dr2_lum, distance]:
                l.append(np.nan)
            continue

        if len(r) > 1:
            print('\tmatched {:d} within 2arcsec, skipping'.format(len(r)))
            for l in [sep, gaia_id, parallax, gmag, dr2_teff, dr2_radius,
                      dr2_lum, distance]:
                l.append(np.nan)
            continue

        sep.append( float((r['dist']*units.deg).to(units.arcsec).value) )
        gaia_id.append( int(r['source_id']) )
        plx = float(r['parallax']*1e-3)
        parallax.append( plx )
        distance.append( 1/plx )
        gmag.append( float(r['phot_g_mean_mag']) )
        dr2_teff.append( float(r['teff_val']) )
        dr2_radius.append( float(r['radius_val']) )
        dr2_lum.append( float(r['lum_val']) )

    sep, gaia_id, parallax, gmag, dr2_teff, dr2_radius, dr2_lum, distance = (
        nparr(sep), nparr(gaia_id), nparr(parallax), nparr(gmag),
        nparr(dr2_teff), nparr(dr2_radius), nparr(dr2_lum), nparr(distance)
    )

    df['sep'] = sep
    df['gaia_id'] = gaia_id
    df['parallax'] = parallax
    df['gmag'] = gmag
    df['dr2_teff'] = dr2_teff
    df['dr2_radius'] = dr2_radius
    df['dr2_lum'] = dr2_lum
    df['distance'] = distance

    savname = '../data/bonomo_HJs_plus_DR2.csv'
    df.to_csv(savname, index=False)
    print('made {}'.format(savname))


def plot_HJs_HR_diagram(catpath, ax):

    df = pd.read_csv(catpath)

    dist_pc = nparr(df['distance'])
    mu = 5 * np.log10(dist_pc)  - 5
    gmag_apparent = nparr(df['gmag'])
    gmag_absolute = gmag_apparent - mu
    df['abs_gmag'] = gmag_absolute

    forbit = nparr(df['forbit'])
    is_ecc = (forbit == 'E')

    planets = nparr(df['planet'])
    # my planet list:
    plnames=['WASP-4b','WASP-5b','WASP-6b','WASP-12b','WASP-18b','WASP-46b']
    if not len(np.setdiff1d(plnames, planets))==0:
        print('error: need all planets to be in B+17 table')
        print(np.setdiff1d(plnames, planets))
        raise AssertionError

    # begin plotting
    x = nparr(df['Teff'])
    y = nparr(df['abs_gmag'])

    ax.scatter(x, y, marker='s', s=5, zorder=2,
               facecolors='none', edgecolors='k', lw=0.4, alpha=0.3)

    marker = itertools.cycle(('o', 'v', '>', 'd', 's', 'P'))
    for tesspl in plnames:
        if tesspl=='WASP-4b':
            s=56
        else:
            s=28
        ax.scatter(x[planets==tesspl], y[planets==tesspl],
                   marker=next(marker), s=s, zorder=3, label=tesspl,
                   edgecolor='black', lw=0.4)

    leg = ax.legend(loc='lower left', fontsize='x-small', framealpha=1)

    ax.set_ylabel('absolute G mag (DR2)')
    ax.set_xlabel('$T_{\mathrm{eff}}$ [K]')

    ax.set_xscale('linear')
    ax.set_yscale('linear')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    ax.set_xlim((7200,4300))
    ax.set_ylim((7.5,2))


def plot_all():

    fig, (a0,a1,a2) = plt.subplots(nrows=3, ncols=1, figsize=(4.25,9.5))

    df_b17 = get_bonomo_2017_mergetable()

    plot_ecc_damping_timescale(df_b17, a0, Qplanetprime=1e5)

    plot_orbital_decay_timescale(df_b17, a1, Qstarprime=1e6)

    # populate the absolute G mag column
    catpath = '../data/bonomo_HJs_plus_DR2.csv'
    if not os.path.exists(catpath):
        do_gaia_match(df_b17)

    if os.path.exists(catpath):
        plot_HJs_HR_diagram(catpath, a2)

    fig.tight_layout(h_pad=0, w_pad=0)
    resultspng = '../results/population_vs_wasp4.png'

    fig.savefig(resultspng, dpi=500, bbox_inches='tight')
    print('saved {:s}'.format(resultspng))
    copyfile(resultspng, '../paper/f4.png')
    print('saved ../paper/f4.png')
    fig.savefig('../paper/f4.pdf', bbox_inches='tight')
    print('saved ../paper/f4.pdf')


if __name__=="__main__":
    plot_all()
