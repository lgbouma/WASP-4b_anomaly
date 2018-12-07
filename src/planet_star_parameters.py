# -*- coding: utf-8 -*-
"""
estimate the stellar density, and mass.
"""
from __future__ import division, print_function
import numpy as np

from astropy import units as u, constants as const

# use parameters from Table 4 of Hoyer+2013
i = 88.52*u.deg
a_by_Rs = 5.463
a_by_Rs_perr = 0.025
a_by_Rs_merr = 0.020
P = 1.33823204*u.day

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

# Keivan value
Rstar = 0.893*u.Rsun
Rstar_err = 0.024*u.Rsun

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

Rp_by_Rs = 0.1538   #measured by me, from TESS
Rp_by_Rs_err = 0.0013
Rp = Rp_by_Rs * Rstar
Rp_upper = (Rp_by_Rs + Rp_by_Rs_err)*(Rstar+Rstar_err)
Rp_lower = (Rp_by_Rs - Rp_by_Rs_err)*(Rstar-Rstar_err)

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

i = 88.52*u.deg # Hoyer+ 2013, table 4
i_upper = i + 0.39*u.deg
i_lower = i - 0.26*u.deg
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

#TODO: add errs
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
