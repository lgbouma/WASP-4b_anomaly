# -*- coding: utf-8 -*-
"""
estimate the stellar density, and mass.
"""
from __future__ import division, print_function
import numpy as np
import pickle

from astropy import units as u, constants as const

# Keivan value
Rstar = 0.893*u.Rsun
Rstar_err = 0.024*u.Rsun

# parameters from Table 4 of Hoyer+2013, for comparison
i_hoyer13 = 88.52*u.deg
a_by_Rs_hoyer13 = 5.463
a_by_Rs_perr_hoyer13 = 0.025
a_by_Rs_merr_hoyer13 = 0.020
P_hoyer13 = 1.33823204*u.day
hoyer_params = [i_hoyer13, a_by_Rs_hoyer13, a_by_Rs_perr_hoyer13,
                a_by_Rs_merr_hoyer13, P_hoyer13]

# get parameters from fit to phase-folded lightcurve
picklepath = ('/home/luke/Dropbox/proj/tessorbitaldecay/results/'
              'tess_lightcurve_fit_parameters/402026209/'
              '402026209_phased_mandelagol_fit_empiricalerrs.pickle' )

d = pickle.load(open(picklepath, 'rb'))
i = d['fitinfo']['finalparams']['incl']*u.deg
i_upper = i + 0.39*u.deg

i = 88.52*u.deg # Hoyer+ 2013, table 4
i_upper = i + d['fitinfo']['finalparamerrs']['std_perrs']['incl']*u.deg
i_lower = i - d['fitinfo']['finalparamerrs']['std_merrs']['incl']*u.deg
a_by_Rs = d['fitinfo']['finalparams']['sma']
a_by_Rs_perr = d['fitinfo']['finalparamerrs']['std_perrs']['sma']
a_by_Rs_merr = d['fitinfo']['finalparamerrs']['std_merrs']['sma']
P = d['fitinfo']['finalparams']['period']*u.day

a_by_Rs = d['fitinfo']['finalparams']['sma']
a_by_Rs_perr = d['fitinfo']['finalparamerrs']['std_perrs']['sma']
a_by_Rs_merr = d['fitinfo']['finalparamerrs']['std_merrs']['sma']

Rp_by_Rs = d['fitinfo']['finalparams']['rp']
Rp_by_Rs_err = np.mean(
    [d['fitinfo']['finalparamerrs']['std_perrs']['rp'],
    d['fitinfo']['finalparamerrs']['std_merrs']['rp']
    ]
)
Rp = Rp_by_Rs * Rstar
Rp_upper = (Rp_by_Rs + Rp_by_Rs_err)*(Rstar+Rstar_err)
Rp_lower = (Rp_by_Rs - Rp_by_Rs_err)*(Rstar-Rstar_err)

my_params = [i, a_by_Rs, a_by_Rs_perr, a_by_Rs_merr, P]

paramnames = ['incl','a/Rstar','a/Rstar_perr','a/Rstar_merr','P']

# the period you adopt here does not particularly matter! the one reported in
# the transit time table does. so using my less precise TESS-only BLS period
# (1.33814, vs hoyer+13: 1.338232 d, is fine.

print('-'*42)
for hparam, mparam, pname in zip(hoyer_params, my_params, paramnames):
    print('{:s}: me: {:.5f}, hoyer+13: {:5f}\n\tme-h13: {:.5f}'.
          format(pname, mparam, hparam, mparam-hparam))
print('-'*42)
print('for selected system parameters table:')
# Rp/Rstar, incl, a/Rstar, ulinear, uquad:
params = ['rp','incl','sma','u_linear','u_quad']
precisions = [5, 2, 3, 3, 3]
for param, precision in zip(params, precisions):
    value = d['fitinfo']['finalparams'][param]
    value_perr = d['fitinfo']['finalparamerrs']['std_perrs'][param]
    value_merr = d['fitinfo']['finalparamerrs']['std_merrs'][param]
    outstr = ('{:s}: '.format(param)+
              '{1:.{0}f}'.format(precision, value)+
              ' +{1:.{0}f}'.format(precision, value_perr)+
              ' -{1:.{0}f}'.format(precision, value_merr)
    )
    print(outstr)
print('-'*42)

# eq 30 of Winn (2010).
# note k^3 = 3e-3, and the density of jupiter is about equal to the density of
# the sun, so neglect the k^3*rho_planet term.

rho_star = 3*np.pi/(const.G*P**2) * a_by_Rs**3
print(rho_star.cgs)

rho_star_upper = 3*np.pi/(const.G*P**2) * (a_by_Rs+a_by_Rs_perr)**3
print(rho_star_upper.cgs)

rho_star_lower = 3*np.pi/(const.G*P**2) * (a_by_Rs-a_by_Rs_merr)**3
print(rho_star_lower.cgs)

print(
    'rhostar = {:.3f} +({:.3f}) -({:.3f}) g/cm^3'.format(
        rho_star.cgs.value,
        (rho_star_upper-rho_star).cgs.value,
        (rho_star-rho_star_lower).cgs.value
    )
)

Mstar = (4*np.pi)/3*rho_star*Rstar**3
Mstar_lower = (4*np.pi)/3*rho_star_lower*(Rstar-Rstar_err)**3
Mstar_upper = (4*np.pi)/3*rho_star_upper*(Rstar+Rstar_err)**3

print(
    'Mstar = {:.3f} +({:.3f}) -({:.3f}) Msun'.format(
        Mstar.to(u.Msun).value,
        (Mstar_upper-Mstar).to(u.Msun).value,
        (Mstar-Mstar_lower).to(u.Msun).value,
    )
)

##########################################
# now get R_p

print(
    'Rp = {:.3f} +({:.3f}) -({:.3f}) Rjup'.format(
        Rp.to(u.Rjup).value,
        (Rp_upper-Rp).to(u.Rjup).value,
        (Rp-Rp_lower).to(u.Rjup).value,
    )
)

# semimajor axis
sma = a_by_Rs * Rstar
sma_upper = (a_by_Rs+a_by_Rs_perr) * (Rstar+Rstar_err)
sma_lower = (a_by_Rs-a_by_Rs_merr) * (Rstar-Rstar_err)

print(
    'sma = {:.4f} +({:.4f}) -({:.4f}) AU'.format(
        sma.to(u.AU).value,
        (sma_upper-sma).to(u.AU).value,
        (sma-sma_lower).to(u.AU).value,
    )
)

# now get M_p

K = 241.1 * u.m/u.s
K_perr = 2.8 * u.m/u.s
K_merr = 3.1 * u.m/u.s

sini = float(np.sin(i))
sini_upper = float(np.sin(i_upper))
sini_lower = float(np.sin(i_lower))

# assume e = 0

Mp = K * (const.G)**(-1/2) * sini**(-1) * Mstar**(1/2) * sma**(1/2)

Mp_upper = (
    (K+K_perr) * (const.G)**(-1/2) * sini_upper**(-1) *
    Mstar_upper**(1/2) * sma_upper**(1/2)
)

Mp_lower = (
    (K-K_merr) * (const.G)**(-1/2) * sini_lower**(-1) *
    Mstar_lower**(1/2) * sma_lower**(1/2)
)


print(
    'Mp = {:.3f} +({:.3f}) -({:.3f}) Mjup'.format(
        Mp.to(u.Mjup).value,
        (Mp_upper - Mp).to(u.Mjup).value,
        (Mp - Mp_lower).to(u.Mjup).value,
    )
)


# finally, Rstar/d
F_bol = 2.8e-10 * u.erg/(u.cm**2 * u.s)
F_bol_err = 0.1e-10 * u.erg/(u.cm**2 * u.s)
Teff = 5400 * u.K
Teff_err = 90 * u.K

Rstar_by_dist = (F_bol / (const.sigma_sb * Teff**4 ) )**(1/2)
Rstar_by_dist_upper = ( (F_bol+F_bol_err) / (const.sigma_sb * (Teff-Teff_err)**4 ) )**(1/2)
Rstar_by_dist_lower = ( (F_bol-F_bol_err) / (const.sigma_sb * (Teff+Teff_err)**4 ) )**(1/2)

print(
    'Rs/dist = {:.3e}, +{:.3e} -{:.3e}'.format(
        Rstar_by_dist.cgs,
        (Rstar_by_dist_upper-Rstar_by_dist).cgs,
        (Rstar_by_dist-Rstar_by_dist_lower).cgs,
    )
)

plx = 3.7145e-3 * u.arcsec
plx_offset = 82e-6 * u.arcsec
plx += plx_offset

plx_err = 0.0517e-3 * u.arcsec

dist_pc = (1/plx.value)*u.pc

Rs_calcd = (Rstar_by_dist * dist_pc).to(u.Rsun)

print(
    'dist = {:.2f}'.format(dist_pc)+
    '\nRstar = {:.3f}'.format(Rs_calcd)
)
